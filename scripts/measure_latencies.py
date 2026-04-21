"""
Measure REAL latency for HuggingFace Spaces via gradio_client.

Unlike tag-based heuristics, this script times actual round-trips:
  1. Client(space_id)      — full handshake including cold-start
  2. view_api()            — API fetch from the running server

For Spaces we've recorded actual predict() calls on, we can also use those
timings as ground truth (higher fidelity than view_api).

Writes to each card:
  - measured_latency_s         float  — real wall-clock seconds
  - measurement_timestamp      str    — ISO datetime
  - measurement_method         str    — "gradio_client_init" | "predict_call"
  - _latency_source            str    — "measured" (overrides the inferred one)

Usage:
    # Measure only the 592 API-verified Spaces
    python scripts/measure_latencies.py --api-verified-only

    # Full catalog (WARNING: will take ~30+ minutes, many will be sleeping)
    python scripts/measure_latencies.py

    # Re-measure even if already timed
    python scripts/measure_latencies.py --force
"""

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CARDS_DIR = ROOT / "fixtures" / "cards"


@contextlib.contextmanager
def _silence_stdio():
    """Suppress gradio_client verbose output."""
    devnull_out = io.StringIO()
    devnull_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull_out, devnull_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def card_path(space_id: str) -> Path:
    return CARDS_DIR / (space_id.replace("/", "_") + ".json")


def load_card(space_id: str) -> Optional[Dict[str, Any]]:
    p = card_path(space_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_card(space_id: str, card: Dict[str, Any]) -> None:
    card_path(space_id).write_text(json.dumps(card, indent=2, default=str))


def measure_one(space_id: str) -> Dict[str, Any]:
    """Perform wall-clock measurement for one Space.

    Returns {status, latency_s, method, error}
    """
    from gradio_client import Client

    t_init_start = time.time()
    try:
        with _silence_stdio():
            client = Client(space_id, verbose=False, download_files=False)
        init_time = time.time() - t_init_start
    except Exception as e:
        return {
            "status": "init_failed",
            "error": str(e)[:200],
            "latency_s": None,
        }

    # Time the API fetch
    t_api_start = time.time()
    try:
        with _silence_stdio():
            client.view_api(return_format="dict")
        api_time = time.time() - t_api_start
    except Exception as e:
        return {
            "status": "view_api_failed",
            "error": str(e)[:200],
            "latency_s": round(init_time, 2),  # at least we have init time
            "method": "gradio_client_init_only",
        }

    # Total round-trip latency (connection + API fetch)
    total = init_time + api_time
    return {
        "status": "ok",
        "latency_s": round(total, 2),
        "method": "gradio_client_init_plus_view_api",
        "init_s": round(init_time, 2),
        "api_s": round(api_time, 2),
    }


def process_space(space_id: str, force: bool = False) -> Dict[str, Any]:
    """Measure + update one card."""
    card = load_card(space_id)
    if not card:
        return {"space_id": space_id, "status": "no_card"}

    if not force and card.get("measured_latency_s") is not None:
        return {"space_id": space_id, "status": "skipped_existing"}

    # Skip Spaces known to be sleeping or errored in prior API scrape
    if card.get("_api_status") in ("sleeping", "not_found", "error"):
        return {"space_id": space_id, "status": f"skip_{card['_api_status']}"}

    result = measure_one(space_id)

    # Update card
    if result.get("latency_s") is not None:
        card["measured_latency_s"] = result["latency_s"]
        card["measurement_method"] = result.get("method", "unknown")
        card["measurement_timestamp"] = datetime.now(timezone.utc).isoformat()
        # Override the coarse estimated_latency_s with measured truth
        card["estimated_latency_s"] = result["latency_s"]
        card["_latency_source"] = "measured"

        # Update speed_tier from measurement
        lat = result["latency_s"]
        if lat < 5:
            card["speed_tier"] = "fast"
        elif lat < 20:
            card["speed_tier"] = "medium"
        elif lat < 60:
            card["speed_tier"] = "slow"
        else:
            card["speed_tier"] = "very_slow"
    else:
        card["_latency_source"] = "inferred"
        card["_measurement_error"] = result.get("error", "")[:200]

    save_card(space_id, card)
    return {
        "space_id": space_id,
        "status": result["status"],
        "latency_s": result.get("latency_s"),
    }


def collect_targets(api_verified_only: bool) -> list:
    """Return list of space_ids to measure."""
    ids = []
    for p in sorted(CARDS_DIR.glob("*.json")):
        try:
            card = json.loads(p.read_text())
        except Exception:
            continue
        if api_verified_only:
            if card.get("_api_status") != "ok" and not card.get("_api_verified"):
                continue
        ids.append(card.get("space_id"))
    return [sid for sid in ids if sid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-verified-only", action="store_true",
                        help="Only measure Spaces we already verified as live (fast, ~10 min)")
    parser.add_argument("--force", action="store_true",
                        help="Re-measure even if already timed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    targets = collect_targets(api_verified_only=args.api_verified_only)
    if args.limit:
        targets = targets[:args.limit]

    print(f"Measuring latency for {len(targets)} Spaces ({args.workers} workers)")
    print()

    stats = {"ok": 0, "init_failed": 0, "view_api_failed": 0, "skipped_existing": 0,
             "no_card": 0, "skip_sleeping": 0, "skip_not_found": 0, "skip_error": 0}
    done = 0
    latencies = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_space, sid, args.force): sid for sid in targets}
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
            except Exception as e:
                r = {"status": "error", "error": str(e)}
            status = r.get("status", "error")
            stats[status] = stats.get(status, 0) + 1
            if r.get("latency_s") is not None:
                latencies.append(r["latency_s"])
            if done % 20 == 0 or status == "ok":
                marker = "✓" if status == "ok" else ("z" if "skip" in status else "✗")
                sid = r.get("space_id", "?")
                lat = r.get("latency_s")
                lat_str = f"{lat:.1f}s" if lat else "-"
                elapsed = time.time() - t0
                print(f"  [{done}/{len(targets)}] {marker} {sid:<50} {status:<20} {lat_str}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n=== Summary ({elapsed:.0f}s) ===")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v}")
    if latencies:
        latencies.sort()
        n = len(latencies)
        print(f"\n  Measured latencies (n={n}):")
        print(f"    min:    {latencies[0]:.2f}s")
        print(f"    p25:    {latencies[n // 4]:.2f}s")
        print(f"    median: {latencies[n // 2]:.2f}s")
        print(f"    p75:    {latencies[3 * n // 4]:.2f}s")
        print(f"    max:    {latencies[-1]:.2f}s")


if __name__ == "__main__":
    main()
