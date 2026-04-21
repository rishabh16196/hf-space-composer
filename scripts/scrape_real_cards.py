"""
Scrape REAL card metadata from HuggingFace Hub for all Spaces in the catalog.

For each Space, calls:
  - HfApi.space_info(space_id, full=True)  - fast, no Space-boot required
  - Optionally downloads README.md  - richer description, often documents I/O

Writes one card JSON per Space to fixtures/cards/<space_id>.json.

Features:
  - Preserves existing "real verified" cards (those with input_schema from gradio_client)
    marked with `_verified_live: true`
  - Skips Spaces whose cards were already scraped (marked `_real_scraped: true`)
    unless --force is given
  - Graceful per-Space error handling
  - Progress logging every 50 Spaces

Usage:
    # Scrape cards for all Spaces in catalog (resume-aware)
    python scripts/scrape_real_cards.py

    # Re-scrape everything from scratch
    python scripts/scrape_real_cards.py --force

    # Also download README.md for richer description
    python scripts/scrape_real_cards.py --with-readme

    # Limit for testing
    python scripts/scrape_real_cards.py --limit 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"
CARDS_DIR = ROOT / "fixtures" / "cards"


# ---------------------------------------------------------------------------
# I/O schema heuristics (from tags, since card itself rarely has this)
# ---------------------------------------------------------------------------

def infer_io_schema_from_tags(tags) -> Dict[str, Any]:
    """Infer input/output schemas from tags (pragmatic heuristic)."""
    t = set(s.lower() for s in (tags or []))

    if {"asr", "transcription", "whisper"} & t or ("audio" in t and "speech" in t):
        return {
            "input_schema": {
                "audio_url": {"type": "string", "description": "Audio file URL", "required": True},
                "language": {"type": "string", "description": "ISO lang code (optional)", "required": False},
            },
            "output_schema": {
                "transcript": {"type": "string", "description": "Transcription"},
            },
        }
    if "translation" in t or "nllb" in t:
        return {
            "input_schema": {
                "text": {"type": "string", "description": "Text to translate", "required": True},
                "src_lang": {"type": "string", "description": "Source language", "required": True},
                "tgt_lang": {"type": "string", "description": "Target language", "required": True},
            },
            "output_schema": {"text": {"type": "string"}},
        }
    if "summarization" in t or "summary" in t:
        return {
            "input_schema": {
                "text": {"type": "string", "description": "Text to summarize", "required": True},
                "max_length": {"type": "integer", "required": False},
            },
            "output_schema": {"summary": {"type": "string"}},
        }
    if "captioning" in t or ("vision" in t and "image" in t):
        return {
            "input_schema": {"image_url": {"type": "string", "description": "Image URL", "required": True}},
            "output_schema": {"caption": {"type": "string"}},
        }
    if "ocr" in t:
        return {
            "input_schema": {"image_url": {"type": "string", "description": "Image with text", "required": True}},
            "output_schema": {"extracted_text": {"type": "string"}},
        }
    if "sentiment" in t or "emotion" in t:
        return {
            "input_schema": {"text": {"type": "string", "description": "Text to classify", "required": True}},
            "output_schema": {"label": {"type": "string"}, "score": {"type": "number"}},
        }
    if "diarization" in t:
        return {
            "input_schema": {"audio_url": {"type": "string", "required": True}},
            "output_schema": {"segments": {"type": "list"}},
        }
    if "pdf" in t or "document" in t:
        return {
            "input_schema": {"pdf_url": {"type": "string", "required": True}},
            "output_schema": {"extracted_text": {"type": "string"}},
        }
    if "ner" in t or "entity-extraction" in t:
        return {
            "input_schema": {"text": {"type": "string", "required": True}},
            "output_schema": {
                "persons": {"type": "list"},
                "organizations": {"type": "list"},
                "locations": {"type": "list"},
            },
        }
    if "code" in t or "coder" in t:
        return {
            "input_schema": {"code_snippet": {"type": "string", "required": True}},
            "output_schema": {"explanation": {"type": "string"}},
        }
    if "tts" in t or "speech-synthesis" in t:
        return {
            "input_schema": {
                "text": {"type": "string", "required": True},
                "voice": {"type": "string", "required": False},
            },
            "output_schema": {"audio_url": {"type": "string"}},
        }
    if "text-to-image" in t or "image-generation" in t or "diffusion" in t:
        return {
            "input_schema": {"prompt": {"type": "string", "required": True}},
            "output_schema": {"image_url": {"type": "string"}},
        }
    if "chatbot" in t or "conversational" in t:
        return {
            "input_schema": {"message": {"type": "string", "required": True}},
            "output_schema": {"response": {"type": "string"}},
        }

    # Default
    return {
        "input_schema": {"input": {"type": "string", "description": "Input", "required": True}},
        "output_schema": {"output": {"type": "string"}},
    }


# ---------------------------------------------------------------------------
# Card fetching
# ---------------------------------------------------------------------------

def card_path(space_id: str) -> Path:
    return CARDS_DIR / (space_id.replace("/", "_") + ".json")


def load_existing_card(space_id: str) -> Optional[Dict[str, Any]]:
    p = card_path(space_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fetch_real_card(
    api: HfApi,
    space_id: str,
    entry: Dict[str, Any],
    with_readme: bool = False,
) -> Optional[Dict[str, Any]]:
    """Fetch real card metadata from HF Hub."""
    try:
        info = api.space_info(space_id)
    except Exception as e:
        return {
            "space_id": space_id,
            "_fetch_error": str(e)[:200],
        }

    card_data = getattr(info, "cardData", None) or {}

    # Real description: try cardData short_description first, then api description,
    # then fall back to catalog summary
    description = (
        card_data.get("short_description")
        or card_data.get("title")
        or getattr(info, "description", "")
        or entry.get("summary", "")
        or space_id
    )
    if isinstance(description, list):
        description = " ".join(str(x) for x in description)
    description = str(description)[:1500]

    # Tags
    tags = list(getattr(info, "tags", []) or [])
    if not tags:
        tags = list(entry.get("tags", []) or [])

    # License
    license_str = str(card_data.get("license", "") or "unknown")

    # SDK
    sdk = getattr(info, "sdk", None) or entry.get("sdk", "unknown")

    # Hardware
    hardware = str(getattr(info, "hardware", "") or card_data.get("sdk_version", ""))

    # Likes
    likes = int(getattr(info, "likes", 0) or entry.get("likes", 0) or 0)

    # Last modified
    last_modified = str(getattr(info, "last_modified", "") or entry.get("last_modified", ""))

    # Title / emoji (HF Space YAML fields)
    title = str(card_data.get("title", "") or space_id.split("/")[-1])
    emoji = str(card_data.get("emoji", "") or "")

    readme_excerpt = ""
    if with_readme:
        try:
            p = hf_hub_download(repo_id=space_id, repo_type="space", filename="README.md")
            txt = Path(p).read_text(errors="ignore")
            # Strip YAML frontmatter
            if txt.startswith("---"):
                parts = txt.split("---", 2)
                if len(parts) >= 3:
                    txt = parts[2]
            readme_excerpt = txt.strip()[:1500]
        except Exception:
            pass

    # Inferred I/O schema
    io = infer_io_schema_from_tags(tags)

    card = {
        "space_id": space_id,
        "title": title,
        "emoji": emoji,
        "description": description,
        "input_schema": io["input_schema"],
        "output_schema": io["output_schema"],
        "license": license_str,
        "hardware": hardware,
        "sdk": sdk,
        "tags": tags,
        "likes": likes,
        "last_modified": last_modified,
        "readme_excerpt": readme_excerpt,
        "_real_scraped": True,
    }
    return card


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if card was already scraped")
    parser.add_argument("--with-readme", action="store_true",
                        help="Also download README.md for richer description")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scrape first N Spaces (for testing)")
    parser.add_argument("--skip-verified", action="store_true",
                        help="Skip Spaces whose card was verified via gradio_client")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Sleep between API calls (seconds, default 0.05)")
    args = parser.parse_args()

    if not CATALOG_FILE.exists():
        print(f"Catalog not found: {CATALOG_FILE}")
        sys.exit(1)

    catalog = json.loads(CATALOG_FILE.read_text())
    if args.limit:
        catalog = catalog[:args.limit]

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    print(f"Authenticated as: {api.whoami().get('name')}")
    print(f"Processing {len(catalog)} Spaces (with-readme={args.with_readme})\n")

    stats = {"scraped": 0, "skipped_existing": 0, "skipped_verified": 0, "errors": 0, "no_card": 0}
    t0 = time.time()

    for i, entry in enumerate(catalog):
        space_id = entry["space_id"]

        # Check existing card
        existing = load_existing_card(space_id)

        # Skip if already scraped and not forced
        if existing and existing.get("_real_scraped") and not args.force:
            stats["skipped_existing"] += 1
            continue

        # Skip verified-live cards (from record_real.py with real gradio_client schemas)
        # These have an 'endpoint' field from view_api
        if args.skip_verified and existing and existing.get("endpoint"):
            stats["skipped_verified"] += 1
            continue

        # Fetch
        card = fetch_real_card(api, space_id, entry, with_readme=args.with_readme)

        if card is None:
            stats["no_card"] += 1
            continue
        if card.get("_fetch_error"):
            stats["errors"] += 1
            # Still write error card so we don't re-try forever
            card_path(space_id).write_text(json.dumps(card, indent=2, default=str))
            continue

        # Preserve certain fields if existing card was manually crafted
        if existing and existing.get("endpoint"):
            # verified card - keep its endpoint + exact I/O schemas
            card["endpoint"] = existing["endpoint"]
            if existing.get("input_schema"):
                card["input_schema"] = existing["input_schema"]
            if existing.get("output_schema"):
                card["output_schema"] = existing["output_schema"]
            card["_verified_live"] = True

        card_path(space_id).write_text(json.dumps(card, indent=2, default=str))
        stats["scraped"] += 1

        # Progress log
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(catalog) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(catalog)}] scraped={stats['scraped']} "
                  f"skipped={stats['skipped_existing']+stats['skipped_verified']} "
                  f"errors={stats['errors']} "
                  f"({rate:.1f} Spaces/s, ~{remaining:.0f}s remaining)")

        if args.sleep > 0:
            time.sleep(args.sleep)

    elapsed = time.time() - t0
    print(f"\n=== Summary ({elapsed:.0f}s) ===")
    print(f"  Scraped fresh:    {stats['scraped']}")
    print(f"  Skipped existing: {stats['skipped_existing']}")
    print(f"  Skipped verified: {stats['skipped_verified']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  No card data:     {stats['no_card']}")


if __name__ == "__main__":
    main()
