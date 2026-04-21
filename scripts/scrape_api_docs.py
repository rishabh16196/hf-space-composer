"""
Scrape REAL API documentation for Spaces via gradio_client.view_api().

For each Space:
  1. Attempt gradio_client.Client(space_id) with a short timeout
  2. If successful, call view_api() to get named_endpoints with parameters/returns
  3. Merge real API schema into the Space's card JSON

Running Spaces: we get real input/output schemas, endpoint names.
Sleeping Spaces: we skip with a 'sleeping' marker so we don't re-try forever.

Parallel workers speed this up (default 8).

Usage:
    # Try all 1500 with 8 parallel workers (~5-10 min)
    python scripts/scrape_api_docs.py

    # Just the Spaces in tasks.json gold pipelines
    python scripts/scrape_api_docs.py --gold-only

    # Limit for testing
    python scripts/scrape_api_docs.py --limit 20

    # Force re-scrape
    python scripts/scrape_api_docs.py --force
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradio_client import Client  # noqa: E402


@contextlib.contextmanager
def _silence_stdio():
    """Suppress noisy gradio_client printing during Client init + view_api."""
    devnull_out = io.StringIO()
    devnull_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull_out, devnull_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"
TASKS_FILE = ROOT / "fixtures" / "tasks.json"
CARDS_DIR = ROOT / "fixtures" / "cards"


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


def gold_pipeline_spaces() -> Set[str]:
    """Return set of Space IDs referenced in any task's gold_pipeline."""
    if not TASKS_FILE.exists():
        return set()
    tasks = json.loads(TASKS_FILE.read_text())
    ids = set()
    for t in tasks:
        for step in t.get("gold_pipeline", []):
            if step.get("space_id"):
                ids.add(step["space_id"])
    return ids


def extract_api_docs(space_id: str, timeout_s: float = 20.0) -> Dict[str, Any]:
    """Try to fetch real API docs via gradio_client.view_api().

    Returns:
        {
          "status": "ok" | "sleeping" | "not_found" | "error",
          "endpoints": [{name, parameters, returns}] if ok,
          "error": str if error,
        }
    """
    try:
        with _silence_stdio():
            client = Client(space_id, verbose=False, download_files=False)
    except Exception as e:
        msg = str(e)
        if "RUNTIME_ERROR" in msg or "invalid state" in msg:
            return {"status": "sleeping", "error": "Space is in RUNTIME_ERROR state"}
        if "404" in msg:
            return {"status": "not_found", "error": "404"}
        if "No such file" in msg or "timeout" in msg.lower():
            return {"status": "error", "error": msg[:200]}
        return {"status": "error", "error": msg[:200]}

    try:
        with _silence_stdio():
            api = client.view_api(return_format="dict")
    except Exception as e:
        return {"status": "error", "error": f"view_api failed: {str(e)[:150]}"}

    endpoints = []
    for name, sig in (api.get("named_endpoints") or {}).items():
        params = []
        for p in sig.get("parameters", []):
            params.append({
                "name": p.get("parameter_name") or p.get("label"),
                "label": p.get("label"),
                "type": str(p.get("python_type", {}).get("type", ""))[:200],
                "component": p.get("component"),
            })
        returns = []
        for p in sig.get("returns", []):
            returns.append({
                "label": p.get("label"),
                "type": str(p.get("python_type", {}).get("type", ""))[:200],
                "component": p.get("component"),
            })
        endpoints.append({
            "name": name,
            "parameters": params,
            "returns": returns,
        })

    return {
        "status": "ok",
        "endpoints": endpoints,
        "n_endpoints": len(endpoints),
    }


def to_input_schema(api_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert gradio params into our card's input_schema format."""
    schema: Dict[str, Any] = {}
    for p in api_params:
        name = p.get("name") or f"param_{len(schema)}"
        schema[name] = {
            "type": p.get("type", "string")[:100],
            "description": p.get("label", ""),
            "required": True,  # gradio doesn't expose required flag reliably
            "component": p.get("component", ""),
        }
    return schema


def to_output_schema(api_returns: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {}
    for i, p in enumerate(api_returns):
        name = p.get("label") or f"output_{i}"
        if name in schema:
            name = f"{name}_{i}"
        schema[name] = {
            "type": p.get("type", "string")[:100],
            "description": p.get("label", ""),
            "component": p.get("component", ""),
        }
    return schema


def process_space(space_id: str, force: bool = False) -> Dict[str, Any]:
    """Process one Space: fetch API, merge into card, return status."""
    card = load_card(space_id)
    if not card:
        return {"space_id": space_id, "status": "no_card"}

    # Skip if already processed (unless forced)
    if not force and card.get("_api_scraped"):
        return {"space_id": space_id, "status": "skipped"}

    result = extract_api_docs(space_id)
    status = result["status"]

    # Persist status in the card to avoid re-trying sleeping/broken Spaces
    card["_api_scraped"] = True
    card["_api_status"] = status

    if status == "ok":
        endpoints = result["endpoints"]
        card["endpoints"] = endpoints
        # Use first named endpoint as primary contract
        if endpoints:
            first = endpoints[0]
            card["endpoint"] = first["name"]
            card["input_schema"] = to_input_schema(first["parameters"])
            card["output_schema"] = to_output_schema(first["returns"])
        card["_api_verified"] = True
    else:
        card["_api_error"] = result.get("error", "")[:200]

    save_card(space_id, card)
    return {"space_id": space_id, "status": status, "n_endpoints": result.get("n_endpoints", 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-scrape even if already tried")
    parser.add_argument("--gold-only", action="store_true",
                        help="Only scrape Spaces in task gold_pipelines")
    parser.add_argument("--limit", type=int, default=None, help="First N Spaces only")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default 8)")
    args = parser.parse_args()

    # Determine target list
    if args.gold_only:
        target_ids = sorted(gold_pipeline_spaces())
        print(f"Mode: gold-only ({len(target_ids)} Spaces)")
    else:
        catalog = json.loads(CATALOG_FILE.read_text())
        target_ids = [e["space_id"] for e in catalog]
        print(f"Mode: full catalog ({len(target_ids)} Spaces)")

    if args.limit:
        target_ids = target_ids[:args.limit]

    # Dispatch to workers
    t0 = time.time()
    stats = {"ok": 0, "sleeping": 0, "not_found": 0, "error": 0, "skipped": 0, "no_card": 0}
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_space, sid, args.force): sid for sid in target_ids}
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
            except Exception as e:
                result = {"status": "error", "error": str(e)[:100]}
            status = result.get("status", "error")
            stats[status] = stats.get(status, 0) + 1

            if done % 50 == 0 or status == "ok":
                sid = result.get("space_id", "?")
                marker = "✓" if status == "ok" else ("z" if status == "sleeping" else "✗")
                nep = result.get("n_endpoints", 0)
                elapsed = time.time() - t0
                print(f"  [{done}/{len(target_ids)}] {marker} {sid:<50} {status}"
                      + (f" ({nep} endpoints)" if nep else "")
                      + f"  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n=== Summary ({elapsed:.0f}s) ===")
    print(f"  ✓ ok (live):      {stats.get('ok', 0)}")
    print(f"  z sleeping:       {stats.get('sleeping', 0)}")
    print(f"  ✗ not_found:      {stats.get('not_found', 0)}")
    print(f"  ✗ error:          {stats.get('error', 0)}")
    print(f"  - skipped:        {stats.get('skipped', 0)}")
    print(f"  - no_card:        {stats.get('no_card', 0)}")


if __name__ == "__main__":
    main()
