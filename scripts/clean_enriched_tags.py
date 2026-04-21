"""
Clean up tags that were wrongly added by the previous (substring-bugged) run
of enrich_tags.py.

The prior run matched keywords as substrings, so e.g. 'ner' matched inside
'inner'/'corner'/'winner', adding NER tags to completely unrelated Spaces.

This script re-computes which enriched tags would be added under the FIXED
(word-boundary) logic, and removes any enriched-tag that's no longer justified.

It only touches tags that appear in enrich_tags.PATTERN_TAGS values — original
tags from HF Hub are preserved.

Run: python scripts/clean_enriched_tags.py [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the patterns from enrich_tags
from scripts.enrich_tags import PATTERN_TAGS

CARDS_DIR = ROOT / "fixtures" / "cards"
CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"

# All tags that the enrichment script can possibly add
ALL_ENRICHED_TAGS: Set[str] = set()
for tags in PATTERN_TAGS.values():
    for t in tags:
        ALL_ENRICHED_TAGS.add(t.lower())


def legitimate_tags_word_boundary(space_id: str, description: str) -> Set[str]:
    """Compute the set of tags that SHOULD be present per word-boundary logic."""
    haystack = (space_id + " " + (description or "")).lower()
    justified: Set[str] = set()
    for keyword, tags in PATTERN_TAGS.items():
        pattern = r"(?:^|[\s\-\_/])%s(?:[\s\-\_/]|$)" % re.escape(keyword)
        if re.search(pattern, haystack):
            for t in tags:
                justified.add(t.lower())
    return justified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cleaned_cards = 0
    total_tags_removed = 0

    for card_path in sorted(CARDS_DIR.glob("*.json")):
        try:
            card = json.loads(card_path.read_text())
        except Exception:
            continue
        current_tags = [t for t in (card.get("tags") or [])]
        current_lower = set(t.lower() for t in current_tags)

        sid = card.get("space_id", "")
        desc = card.get("description") or ""
        justified = legitimate_tags_word_boundary(sid, desc)

        # A tag is "wrongly enriched" if:
        #   - it is in the enriched-tag universe (PATTERN_TAGS values)
        #   - AND it is currently present
        #   - AND it is NOT in the justified set
        to_remove = (ALL_ENRICHED_TAGS & current_lower) - justified

        if not to_remove:
            continue

        new_tags = [t for t in current_tags if t.lower() not in to_remove]
        card["tags"] = new_tags
        total_tags_removed += len(to_remove)
        cleaned_cards += 1

        if not args.dry_run:
            card_path.write_text(json.dumps(card, indent=2, default=str))

    print(f"Cleaned {cleaned_cards} cards, removed {total_tags_removed} wrongly-enriched tags")

    # Re-sync catalog
    if not args.dry_run and CATALOG_FILE.exists():
        catalog = json.loads(CATALOG_FILE.read_text())
        synced = 0
        for entry in catalog:
            safe = entry.get("space_id", "").replace("/", "_")
            p = CARDS_DIR / f"{safe}.json"
            if not p.exists():
                continue
            card = json.loads(p.read_text())
            if set(t.lower() for t in card.get("tags") or []) != set(t.lower() for t in entry.get("tags") or []):
                entry["tags"] = card.get("tags") or []
                synced += 1
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2, default=str))
        print(f"Re-synced {synced} catalog entries")


if __name__ == "__main__":
    main()
