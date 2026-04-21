"""
Discover real, callable HuggingFace Spaces for our task domains.

For each domain (audio, vision, document, code, multimodal), search HF Hub for
top Spaces, filter to gradio-based ones, and try instantiating gradio_client
to verify they're callable.

Outputs:
  - Prints a ranked candidate list per domain
  - Optionally saves a curated catalog to fixtures/space_catalog_real.json

Usage:
    # Just discover and print
    python scripts/discover_real_spaces.py --domain audio

    # Discover all domains
    python scripts/discover_real_spaces.py --all

    # Discover and save shortlist
    python scripts/discover_real_spaces.py --all --save
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

FIXTURES_DIR = ROOT / "fixtures"


# Search queries per domain. Multiple queries widen coverage.
DOMAIN_SEARCHES: Dict[str, List[Dict[str, Any]]] = {
    "audio": [
        {"query": "whisper"},
        {"query": "speech recognition"},
        {"query": "audio transcribe"},
        {"query": "ASR multilingual"},
    ],
    "translation": [
        {"query": "translate"},
        {"query": "nllb"},
        {"query": "marian translate"},
        {"query": "opus mt"},
    ],
    "summarization": [
        {"query": "summarize"},
        {"query": "BART summary"},
        {"query": "T5 summary"},
    ],
    "vision_caption": [
        {"query": "image to text"},
        {"query": "BLIP caption"},
        {"query": "image describe"},
        {"query": "vision language"},
    ],
    "vision_ocr": [
        {"query": "OCR"},
        {"query": "trocr"},
        {"query": "EasyOCR"},
    ],
    "sentiment": [
        {"query": "sentiment"},
        {"query": "emotion"},
        {"query": "twitter roberta"},
    ],
    "code": [
        {"query": "code"},
        {"query": "starcoder"},
        {"query": "code llama"},
    ],
    "tts": [
        {"query": "text to speech"},
        {"query": "TTS"},
        {"query": "voice clone"},
    ],
}


def search_spaces(api: HfApi, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Search HF Spaces by query. Returns list of dicts."""
    try:
        spaces = list(api.list_spaces(search=query, limit=top_k, full=True))
    except Exception as e:
        print(f"  ! Search '{query}' failed: {e}")
        return []

    results = []
    for s in spaces:
        results.append({
            "space_id": s.id,
            "author": s.author or s.id.split("/")[0],
            "likes": getattr(s, "likes", 0) or 0,
            "sdk": getattr(s, "sdk", "unknown"),
            "tags": list(getattr(s, "tags", []) or []),
            "private": getattr(s, "private", False),
            "last_modified": str(getattr(s, "last_modified", "")),
        })
    return results


def filter_gradio_public(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only gradio-based, public Spaces."""
    return [
        r for r in results
        if r.get("sdk") == "gradio"
        and not r.get("private")
        and r.get("likes", 0) > 0
    ]


def try_instantiate_client(space_id: str, timeout_s: float = 15.0) -> Optional[Dict[str, Any]]:
    """Attempt to construct a gradio_client.Client for this Space.

    Returns dict with 'api_info' if successful, None if failed.
    """
    try:
        from gradio_client import Client
    except ImportError:
        return None

    start = time.time()
    try:
        # Note: gradio_client.Client init can be slow (validates the API)
        client = Client(space_id, verbose=False)
        api_info = client.view_api(return_format="dict")
        elapsed = time.time() - start
        endpoints = list(api_info.get("named_endpoints", {}).keys())
        return {
            "callable": True,
            "init_seconds": round(elapsed, 1),
            "endpoints": endpoints[:5],
            "n_endpoints": len(endpoints),
        }
    except Exception as e:
        return {
            "callable": False,
            "error": str(e)[:150],
            "init_seconds": round(time.time() - start, 1),
        }


def discover_domain(api: HfApi, domain: str, verify_callable: bool = True, max_per_query: int = 10):
    """Discover Spaces for one domain."""
    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain}")
    print(f"{'=' * 70}")

    seen = set()
    all_candidates: List[Dict[str, Any]] = []

    for search in DOMAIN_SEARCHES[domain]:
        query = search["query"]
        print(f"\n  > Search: '{query}'")
        results = search_spaces(api, query, top_k=max_per_query)
        gradio_results = filter_gradio_public(results)
        print(f"    {len(results)} results, {len(gradio_results)} gradio public")
        for r in gradio_results:
            if r["space_id"] in seen:
                continue
            seen.add(r["space_id"])
            all_candidates.append(r)

    # Sort by likes desc
    all_candidates.sort(key=lambda x: x.get("likes", 0), reverse=True)
    top = all_candidates[:5]  # Top 5 to verify

    print(f"\n  Top {len(top)} candidates by likes:")
    for r in top:
        print(f"    {r['space_id']:<55} likes={r['likes']:<5} sdk={r['sdk']}")

    if not verify_callable:
        return top

    # Verify each is callable
    print(f"\n  Verifying gradio_client connectivity...")
    verified = []
    for r in top:
        print(f"    Trying {r['space_id']}...")
        api_info = try_instantiate_client(r["space_id"])
        r["client_check"] = api_info
        if api_info and api_info.get("callable"):
            print(f"      ✓ callable ({api_info['init_seconds']}s, {api_info['n_endpoints']} endpoints)")
            verified.append(r)
        else:
            err = (api_info or {}).get("error", "unknown")
            print(f"      ✗ not callable: {err[:80]}")

    print(f"\n  ✓ {len(verified)}/{len(top)} Spaces are callable via gradio_client")
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(DOMAIN_SEARCHES.keys()) + ["all"], default="all")
    parser.add_argument("--all", dest="all_domains", action="store_true", help="Alias for --domain all")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip gradio_client connectivity check (faster)")
    parser.add_argument("--save", action="store_true",
                        help="Save discovered Spaces to fixtures/space_catalog_real.json")
    parser.add_argument("--max-per-query", type=int, default=10)
    args = parser.parse_args()

    api = HfApi()
    print(f"Using HF API as: {api.whoami()['name']}")

    if args.all_domains:
        args.domain = "all"
    domains = list(DOMAIN_SEARCHES.keys()) if args.domain == "all" else [args.domain]
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for d in domains:
        results = discover_domain(api, d, verify_callable=not args.no_verify, max_per_query=args.max_per_query)
        all_results[d] = results

    if args.save:
        out_path = FIXTURES_DIR / "space_catalog_real.json"
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n✓ Saved to {out_path}")

    print("\n=== Summary ===")
    for d, results in all_results.items():
        callable_count = sum(1 for r in results if (r.get("client_check") or {}).get("callable"))
        print(f"  {d:<20} {len(results):<3} candidates ({callable_count} callable)")


if __name__ == "__main__":
    main()
