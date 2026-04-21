"""
Enrich card tags by inferring domain tags from space_id + description.

Many HF Spaces have sparse tags (e.g., just 'gradio', 'region:us'). For our
Auditor's faster-equivalent detection to work, we need semantic domain tags.

This script adds missing domain tags by:
  1. Matching keywords in space_id (e.g., 'FLUX' → text-to-image, diffusion)
  2. Matching keywords in description
  3. Preserving all existing tags

Run: python scripts/enrich_tags.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "fixtures" / "cards"
CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"


# Keyword → tags to add (case-insensitive, matches in id or description)
PATTERN_TAGS = {
    # Image generation
    "flux":           ["text-to-image", "image-generation", "diffusion", "vision"],
    "stable-diffusion": ["text-to-image", "image-generation", "diffusion", "vision"],
    "sdxl":           ["text-to-image", "image-generation", "diffusion", "vision"],
    "animagine":      ["text-to-image", "image-generation", "diffusion", "vision"],
    "illusion":       ["text-to-image", "image-generation", "diffusion", "vision"],
    "z-image":        ["text-to-image", "image-generation", "diffusion", "vision"],
    "qr-code":        ["text-to-image", "image-generation", "vision"],
    "qr_code":        ["text-to-image", "image-generation", "vision"],
    "biggan":         ["text-to-image", "image-generation", "vision"],
    "image-edit":     ["image-generation", "vision"],
    "magic-quill":    ["image-generation", "vision"],
    "image-enhancer": ["image-generation", "vision"],
    "redux":          ["text-to-image", "image-generation", "diffusion", "vision"],
    # Audio
    "whisper":        ["audio", "asr", "transcription"],
    "wav2vec":        ["audio", "asr", "transcription"],
    "edge-tts":       ["audio", "tts", "speech-synthesis"],
    "xtts":           ["audio", "tts", "speech-synthesis"],
    "dia-1.6":        ["audio", "tts", "speech-synthesis"],
    "kokoro":         ["audio", "tts", "speech-synthesis"],
    "tts":            ["audio", "tts", "speech-synthesis"],
    "speech":         ["audio"],
    "song":           ["audio", "music", "song-generation"],
    # Translation / NLP
    "nllb":           ["translation", "multilingual", "text"],
    "opus-mt":        ["translation", "text"],
    "bart":           ["summarization", "text"],
    "t5":             ["text"],
    "bert-ner":       ["ner", "entity-extraction", "text"],
    # Vision
    "blip":           ["vision", "captioning", "image-to-text"],
    "trocr":          ["vision", "ocr"],
    "easyocr":        ["vision", "ocr"],
    "deepseek-ocr":   ["vision", "ocr"],
    "surya-ocr":      ["vision", "ocr"],
    "depth":          ["vision", "depth-estimation"],
    "detectron":      ["vision"],
    "pose":           ["vision"],
    "yolo":           ["vision", "object-detection"],
    # Code
    "code":           ["code", "code-generation"],
    "coder":          ["code", "code-generation"],
    "starcoder":      ["code", "code-generation"],
    "codeformer":     ["code"],
    # 3D
    "hunyuan3d":      ["3d", "text-to-3d", "vision"],
    "trellis":        ["3d", "text-to-3d", "vision"],
    "wan2":           ["video", "video-generation", "vision"],
    "wan-ai":         ["video", "video-generation", "vision"],
    # Other
    "leaderboard":    ["leaderboard", "benchmark"],
    "arena":          ["leaderboard", "benchmark"],
    "face-swap":      ["vision", "video"],
    "inpaint":        ["image-generation", "vision"],
    "caption":        ["vision", "captioning", "image-to-text"],
    "diffusers":      ["diffusion", "image-generation"],
    "sentiment":      ["sentiment", "text"],
    "emotion":        ["sentiment", "text"],
    "ner":            ["ner", "entity-extraction", "text"],
    "prompt":         ["text"],
    "translate":      ["translation", "text"],
    "summarize":      ["summarization", "text"],
    "music":          ["audio", "music"],
}


def infer_extra_tags(space_id: str, description: str, existing_tags: List[str]) -> Set[str]:
    """Return tags to add based on id/description keyword matching.

    Uses word-boundary matching (not substring) to avoid false positives like
    'ner' matching 'inner' or 'corner'.
    """
    import re
    haystack = (space_id + " " + (description or "")).lower()
    existing = set(t.lower() for t in existing_tags)
    new_tags: Set[str] = set()

    for keyword, tags in PATTERN_TAGS.items():
        # Word boundary match: keyword must be a whole word or id-segment
        # Allow hyphens, slashes, underscores as word-boundary substitutes
        pattern = r"(?:^|[\s\-\_/])%s(?:[\s\-\_/]|$)" % re.escape(keyword)
        if re.search(pattern, haystack):
            for t in tags:
                if t.lower() not in existing:
                    new_tags.add(t)
    return new_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CARDS_DIR.exists():
        print(f"No cards dir: {CARDS_DIR}")
        sys.exit(1)

    updated = 0
    total = 0
    total_new_tags = 0

    for card_path in sorted(CARDS_DIR.glob("*.json")):
        try:
            card = json.loads(card_path.read_text())
        except Exception:
            continue
        total += 1
        sid = card.get("space_id", "")
        desc = card.get("description") or ""
        existing = list(card.get("tags") or [])
        new = infer_extra_tags(sid, desc, existing)
        if not new:
            continue
        card["tags"] = existing + sorted(new)
        card["_tags_enriched"] = True
        total_new_tags += len(new)
        if not args.dry_run:
            card_path.write_text(json.dumps(card, indent=2, default=str))
        updated += 1

    print(f"Enriched {updated}/{total} cards with {total_new_tags} new tags")

    # Also re-merge into catalog
    if not args.dry_run and CATALOG_FILE.exists():
        catalog = json.loads(CATALOG_FILE.read_text())
        catalog_updated = 0
        for entry in catalog:
            sid = entry.get("space_id", "")
            safe = sid.replace("/", "_")
            card_path = CARDS_DIR / f"{safe}.json"
            if card_path.exists():
                card = json.loads(card_path.read_text())
                entry_tags = set(entry.get("tags") or [])
                card_tags = set(card.get("tags") or [])
                if card_tags - entry_tags:
                    entry["tags"] = sorted(entry_tags | card_tags)
                    catalog_updated += 1
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2, default=str))
        print(f"Re-synced {catalog_updated} catalog entries with enriched tags")


if __name__ == "__main__":
    main()
