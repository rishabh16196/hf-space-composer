"""
Verify a shortlist of real HF Spaces by attempting to call them via gradio_client.

For each Space:
  1. Construct gradio_client.Client (validates it boots)
  2. Print its API endpoints
  3. (Optional) Test a sample call
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Curated shortlist for the demo
SHORTLIST = [
    "hf-audio/whisper-large-v3",
    "UNESCO/nllb",
    "Alifarsi/news_summarizer",
    "ybelkada/blip-image-captioning-space-large",
    "tomofi/EasyOCR",
    "hexgrad/Kokoro-TTS",
    "innoai/Edge-TTS-Text-to-Speech",
]


def verify(space_id: str, sample_inputs: Dict[str, Any] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """Verify a Space is callable. Returns dict with status."""
    from gradio_client import Client

    result: Dict[str, Any] = {"space_id": space_id}
    print(f"\n=== {space_id} ===")
    start = time.time()
    try:
        client = Client(space_id, verbose=False, download_files=False)
        result["init_seconds"] = round(time.time() - start, 1)
        print(f"  ✓ Client init: {result['init_seconds']}s")

        api_info = client.view_api(return_format="dict")
        endpoints = list(api_info.get("named_endpoints", {}).keys())
        result["endpoints"] = endpoints
        print(f"  Named endpoints: {endpoints[:5]}")

        # Print first endpoint signature
        if endpoints:
            first_ep = endpoints[0]
            sig = api_info["named_endpoints"][first_ep]
            inputs = sig.get("parameters", [])
            outputs = sig.get("returns", [])
            print(f"  First endpoint '{first_ep}':")
            print(f"    inputs:  {[(p.get('label'), p.get('python_type', {}).get('type')) for p in inputs[:5]]}")
            print(f"    outputs: {[(p.get('label'), p.get('python_type', {}).get('type')) for p in outputs[:3]]}")
            result["first_endpoint"] = first_ep
            result["input_signature"] = [
                {"label": p.get("label"), "type": str(p.get("python_type", {}).get("type", ""))[:30]}
                for p in inputs[:5]
            ]

        result["status"] = "ok"
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:300]
        print(f"  ✗ Failed: {result['error'][:100]}")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save verification report")
    parser.add_argument("--spaces", nargs="+", help="Override shortlist with given Space IDs")
    args = parser.parse_args()

    spaces = args.spaces or SHORTLIST
    print(f"Verifying {len(spaces)} Spaces...\n")

    results: List[Dict[str, Any]] = []
    for sid in spaces:
        r = verify(sid)
        results.append(r)

    print("\n=== Summary ===")
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    print(f"  {len(ok)} OK, {len(err)} errors")
    for r in ok:
        print(f"    ✓ {r['space_id']:<55} ({r.get('init_seconds')}s, {len(r.get('endpoints', []))} endpoints)")
    for r in err:
        print(f"    ✗ {r['space_id']:<55} {r.get('error', '')[:60]}")

    if args.save:
        out = ROOT / "fixtures" / "verified_spaces.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()
