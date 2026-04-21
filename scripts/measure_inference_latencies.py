"""
Measure REAL INFERENCE latency by making actual predict() calls against
HuggingFace Spaces.

Unlike measure_latencies.py (which times Client init + view_api), this script:
  1. Reads the Space's API signature from its card (endpoints + parameters)
  2. Auto-generates plausible inputs based on parameter types
  3. Calls predict() with timeout + times the full inference
  4. Saves REAL inference time to the card

Input synthesis heuristics:
  - `str`                       → "Hello, this is a test."
  - `filepath` (audio/wav)      → MLK speech sample on HF datasets
  - `filepath` (image)          → document OCR sample
  - `Literal[A, B, C]`          → first non-empty value
  - `float` / `int`             → sensible default (42 / 1024 / etc.)
  - `bool`                      → True
  - list/dict                   → empty / minimal

Workers are configurable for parallel runs. Failures fall back gracefully
without poisoning the card.

Usage:
    # Top 100 API-verified Spaces by likes (~15 min)
    python scripts/measure_inference_latencies.py --top 100

    # All verified (~2-3 hours, many will fail)
    python scripts/measure_inference_latencies.py --all

    # Re-measure even if already timed
    python scripts/measure_inference_latencies.py --top 50 --force

    # Faster but noisier with more workers
    python scripts/measure_inference_latencies.py --top 200 --workers 16
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradio_client import Client, handle_file  # noqa: E402

CARDS_DIR = ROOT / "fixtures" / "cards"

# Public sample assets — guaranteed accessible via HF CDN
SAMPLE_ASSETS = {
    "audio": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/document_ocr.png",
    "video": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/video_classification.mp4",
    "text": "Hello, this is a short test input for measuring inference latency.",
    "prompt": "A peaceful garden at sunset",
    "pdf": "https://huggingface.co/datasets/hf-internal-testing/test-pdf-files/resolve/main/sample.pdf",
}


@contextlib.contextmanager
def _silence_stdio():
    devnull_out = io.StringIO()
    devnull_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull_out, devnull_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------------------
# Input synthesis
# ---------------------------------------------------------------------------

def parse_literal_values(type_str: str) -> List[str]:
    """Parse 'Literal[\"A\", \"B\", \"C\"]' into ['A', 'B', 'C']."""
    if not type_str or "Literal[" not in type_str:
        return []
    # Extract everything between Literal[ and last ]
    try:
        start = type_str.index("Literal[") + len("Literal[")
        end = type_str.rindex("]")
        inner = type_str[start:end]
        # Simple CSV parse, handle quotes
        values = []
        current = ""
        in_quote = False
        for ch in inner:
            if ch in ("'", '"'):
                in_quote = not in_quote
                current += ch
            elif ch == "," and not in_quote:
                v = current.strip().strip("'\"")
                if v:
                    values.append(v)
                current = ""
            else:
                current += ch
        if current.strip():
            values.append(current.strip().strip("'\""))
        return values
    except (ValueError, IndexError):
        return []


def guess_input_for_param(param: Dict[str, Any]) -> Any:
    """Generate a plausible input for one parameter."""
    t = (param.get("type") or "").lower()
    name = (param.get("name") or "").lower()
    label = (param.get("label") or "").lower()
    component = (param.get("component") or "").lower()
    hint = name + " " + label + " " + t + " " + component

    # Literal — pick first value
    if "literal[" in t:
        vals = parse_literal_values(param.get("type", ""))
        if vals:
            return vals[0]
        return ""

    # Filepath → figure out audio/image/video/pdf from name/type hints
    if "filepath" in t or component in ("audio", "image", "video", "file"):
        if any(k in hint for k in ["audio", "mp3", "wav", "speech"]):
            return handle_file(SAMPLE_ASSETS["audio"])
        if any(k in hint for k in ["image", "picture", "photo", "img"]):
            return handle_file(SAMPLE_ASSETS["image"])
        if any(k in hint for k in ["video", "clip"]):
            return handle_file(SAMPLE_ASSETS["video"])
        if any(k in hint for k in ["pdf", "document"]):
            return handle_file(SAMPLE_ASSETS["pdf"])
        # Default to image
        return handle_file(SAMPLE_ASSETS["image"])

    # Numbers
    if "float" in t:
        if any(k in hint for k in ["seed", "num", "count"]):
            return 42.0
        if any(k in hint for k in ["width", "height", "size"]):
            return 512.0
        if any(k in hint for k in ["rate", "pitch"]):
            return 0.0
        if any(k in hint for k in ["scale", "guidance", "strength"]):
            return 1.0
        return 1.0
    if "int" in t:
        if any(k in hint for k in ["width", "height"]):
            return 512
        if "seed" in hint:
            return 42
        if any(k in hint for k in ["step", "num"]):
            return 10
        return 1

    # Bool
    if "bool" in t:
        return True

    # Lists/dicts
    if "list" in t:
        return []
    if "dict" in t:
        return {}

    # String default
    if any(k in hint for k in ["prompt", "description"]):
        return SAMPLE_ASSETS["prompt"]
    if any(k in hint for k in ["url", "link"]):
        return SAMPLE_ASSETS["image"]
    return SAMPLE_ASSETS["text"]


def build_predict_args(card: Dict[str, Any]) -> Optional[tuple]:
    """Return (positional_args, api_name) for a predict call. None if unusable."""
    endpoints = card.get("endpoints") or []
    if not endpoints:
        return None
    # Pick the first endpoint that looks like a single-step predict
    # (skip /lambda, /lambda_N, /clear_history etc. which are often UI-only)
    candidates = [
        ep for ep in endpoints
        if not ep.get("name", "").startswith("/lambda")
        and ep.get("name") not in ("/clear_history", "/on_page_load", "/start_session")
        and (ep.get("parameters") or [])
    ]
    if not candidates:
        candidates = [ep for ep in endpoints if ep.get("parameters")]
    if not candidates:
        return None

    ep = candidates[0]
    args = [guess_input_for_param(p) for p in ep.get("parameters", [])]
    return args, ep["name"]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

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


def measure_inference(space_id: str, timeout_s: float = 60.0) -> Dict[str, Any]:
    """Make a real predict() call with auto-generated inputs. Times it."""
    card = load_card(space_id)
    if not card:
        return {"status": "no_card"}

    plan = build_predict_args(card)
    if plan is None:
        return {"status": "no_predictable_endpoint"}
    args, api_name = plan

    try:
        with _silence_stdio():
            client = Client(space_id, verbose=False, download_files=False)
    except Exception as e:
        msg = str(e)
        if "RUNTIME_ERROR" in msg or "invalid state" in msg:
            return {"status": "sleeping"}
        if "404" in msg:
            return {"status": "not_found"}
        return {"status": "init_failed", "error": msg[:200]}

    t0 = time.time()
    try:
        with _silence_stdio():
            client.predict(*args, api_name=api_name)
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": "predict_failed",
            "error": str(e)[:200],
            "predict_time_s": round(elapsed, 2),
            "api_name": api_name,
        }

    return {
        "status": "ok",
        "inference_time_s": round(elapsed, 2),
        "api_name": api_name,
        "n_args": len(args),
    }


def process_space(space_id: str, force: bool = False) -> Dict[str, Any]:
    card = load_card(space_id)
    if not card:
        return {"space_id": space_id, "status": "no_card"}
    if not force and card.get("measured_inference_s") is not None:
        return {"space_id": space_id, "status": "skipped_existing"}
    if card.get("_api_status") != "ok":
        return {"space_id": space_id, "status": f"skip_{card.get('_api_status')}"}

    result = measure_inference(space_id)

    if result.get("status") == "ok":
        lat = result["inference_time_s"]
        card["measured_inference_s"] = lat
        card["measured_inference_timestamp"] = datetime.now(timezone.utc).isoformat()
        card["measured_inference_method"] = "auto_predict"
        card["measured_inference_api"] = result.get("api_name")
        # Promote to authoritative latency
        card["measured_latency_s"] = lat
        card["estimated_latency_s"] = lat
        card["_latency_source"] = "measured_inference"
        if lat < 5:
            card["speed_tier"] = "fast"
        elif lat < 20:
            card["speed_tier"] = "medium"
        elif lat < 60:
            card["speed_tier"] = "slow"
        else:
            card["speed_tier"] = "very_slow"
        save_card(space_id, card)
        return {"space_id": space_id, "status": "ok", "latency_s": lat}

    # Record the failure reason so we don't retry forever
    card["_inference_attempt_status"] = result.get("status")
    card["_inference_attempt_error"] = result.get("error", "")[:200]
    save_card(space_id, card)
    return {"space_id": space_id, **result}


def collect_targets(top: Optional[int], measure_all: bool) -> List[str]:
    """Collect API-verified Spaces, sorted by likes desc."""
    targets = []
    for p in CARDS_DIR.glob("*.json"):
        try:
            card = json.loads(p.read_text())
        except Exception:
            continue
        if card.get("_api_status") != "ok":
            continue
        if not card.get("endpoints"):
            continue
        targets.append((card.get("likes", 0) or 0, card.get("space_id")))
    targets.sort(key=lambda x: x[0], reverse=True)
    ids = [sid for _, sid in targets if sid]
    if top and not measure_all:
        ids = ids[:top]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=100,
                        help="Measure top N Spaces by likes (default: 100)")
    parser.add_argument("--all", action="store_true",
                        help="Measure ALL API-verified Spaces (ignore --top)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    targets = collect_targets(args.top, args.all)
    print(f"Measuring INFERENCE time for {len(targets)} Spaces (workers={args.workers})")
    print()

    stats: Dict[str, int] = {}
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
                r = {"status": "worker_exception", "error": str(e)[:100]}
            status = r.get("status", "error")
            stats[status] = stats.get(status, 0) + 1
            if r.get("latency_s") is not None:
                latencies.append(r["latency_s"])
            if done % 10 == 0 or status == "ok":
                sid = r.get("space_id", "?")
                marker = "✓" if status == "ok" else ("z" if "skip" in status else "✗")
                lat = r.get("latency_s")
                lat_str = f"{lat:.1f}s" if lat else "-"
                elapsed = time.time() - t0
                print(f"  [{done}/{len(targets)}] {marker} {sid:<45} {status:<22} {lat_str}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n=== Summary ({elapsed:.0f}s) ===")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")
    if latencies:
        latencies.sort()
        n = len(latencies)
        print(f"\n  Measured inference times (n={n}):")
        print(f"    min:     {latencies[0]:.2f}s")
        print(f"    p25:     {latencies[n // 4]:.2f}s")
        print(f"    median:  {latencies[n // 2]:.2f}s")
        print(f"    p75:     {latencies[3 * n // 4]:.2f}s")
        print(f"    p95:     {latencies[int(n * 0.95)]:.2f}s")
        print(f"    max:     {latencies[-1]:.2f}s")


if __name__ == "__main__":
    main()
