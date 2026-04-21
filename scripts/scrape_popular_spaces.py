"""
Scrape 1000-1500 most-liked public HuggingFace Spaces for the mock catalog.

For each Space, we capture:
  - space_id, author, likes, sdk, tags, summary, last_modified

The scraper:
  1. Uses HfApi.list_spaces with full metadata
  2. Filters to public + callable-ish SDKs (gradio primarily, plus streamlit/static)
  3. Ranks by likes (descending)
  4. Preserves any existing "real" or "decoy" entries in the catalog
  5. Writes merged result to fixtures/space_catalog.json

This gives the env a realistic, large catalog surface for the agent's search()
action — 1500 real Space IDs, real likes, real tags — even though most of them
are never actually called (they just appear in search results).

Usage:
    python scripts/scrape_popular_spaces.py --count 1500
    python scripts/scrape_popular_spaces.py --count 1500 --include-streamlit
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"


def load_existing_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_FILE.exists():
        return []
    with open(CATALOG_FILE) as f:
        return json.load(f)


def extract_preserved(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep real and decoy entries across scrapes."""
    preserved = []
    for s in existing:
        if s.get("_real"):
            preserved.append(s)
        elif "DECOY" in (s.get("summary") or ""):
            preserved.append(s)
    return preserved


def fetch_top_spaces(
    api: HfApi,
    target_count: int,
    allowed_sdks: Set[str],
    page_size: int = 200,
) -> List[Dict[str, Any]]:
    """Fetch up to target_count Spaces ranked by likes (desc).

    HfApi.list_spaces returns an iterator; we pull until we have enough
    after filtering.
    """
    print(f"Fetching top Spaces (target={target_count}, SDKs={allowed_sdks})...")
    collected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    t0 = time.time()

    # Strategy: iterate list_spaces sorted by likes (API default is recent;
    # we need to fetch a large batch and sort locally since sort param varies)
    # Pull 5x the target to leave room for filtering.
    fetch_limit = target_count * 5
    try:
        iterator = api.list_spaces(full=True, limit=fetch_limit)
    except TypeError:
        # Older API
        iterator = api.list_spaces(full=True)

    for i, s in enumerate(iterator):
        if i % 500 == 0 and i > 0:
            print(f"  ...scanned {i} ({len(collected)} kept, {time.time() - t0:.0f}s)")

        space_id = s.id
        if space_id in seen:
            continue
        seen.add(space_id)

        # Skip private Spaces
        if getattr(s, "private", False):
            continue

        sdk = getattr(s, "sdk", None) or "unknown"
        if allowed_sdks and sdk not in allowed_sdks:
            continue

        likes = int(getattr(s, "likes", 0) or 0)

        # Tags
        tags = list(getattr(s, "tags", []) or [])

        # Description - try cardData first, fall back to id-derived
        card_data = getattr(s, "cardData", {}) or {}
        summary = (
            card_data.get("short_description")
            or card_data.get("title")
            or (getattr(s, "description", "") or "")
        )
        if isinstance(summary, list):
            summary = " ".join(str(x) for x in summary)
        summary = str(summary)[:250] if summary else space_id

        entry = {
            "space_id": space_id,
            "name": space_id.split("/")[-1],
            "author": s.author or space_id.split("/")[0],
            "downloads": 0,  # Spaces don't have a downloads counter the same way
            "likes": likes,
            "tags": tags,
            "sdk": sdk,
            "summary": summary,
            "last_modified": str(getattr(s, "last_modified", "")),
        }
        collected.append(entry)

        if len(collected) >= fetch_limit:
            break

    # Sort by likes descending and truncate to target
    collected.sort(key=lambda x: x.get("likes", 0), reverse=True)
    return collected[:target_count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1500,
                        help="Target number of Spaces to scrape (default: 1500)")
    parser.add_argument("--include-streamlit", action="store_true",
                        help="Include Streamlit Spaces (default: gradio + static only)")
    parser.add_argument("--include-all-sdks", action="store_true",
                        help="Include all SDKs (docker, static, etc.)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch but don't write catalog")
    args = parser.parse_args()

    # Determine allowed SDKs
    if args.include_all_sdks:
        allowed_sdks = set()  # empty set = allow all
    elif args.include_streamlit:
        allowed_sdks = {"gradio", "streamlit", "static"}
    else:
        allowed_sdks = {"gradio", "static"}

    api = HfApi()
    whoami = api.whoami()
    print(f"Authenticated as: {whoami.get('name', 'unknown')}\n")

    # Preserve real + decoy entries
    existing = load_existing_catalog()
    preserved = extract_preserved(existing)
    preserved_ids = {p["space_id"] for p in preserved}
    print(f"Preserving {len(preserved)} entries (real + decoy)")
    for p in preserved:
        tag = "[REAL]" if p.get("_real") else "[DECOY]"
        print(f"  {tag} {p['space_id']}")

    # Fetch
    t0 = time.time()
    scraped = fetch_top_spaces(api, args.count, allowed_sdks)
    elapsed = time.time() - t0
    print(f"\nScraped {len(scraped)} Spaces in {elapsed:.0f}s")

    # Filter out any that overlap with preserved (preserve wins)
    deduped = [s for s in scraped if s["space_id"] not in preserved_ids]
    print(f"After dedup vs preserved: {len(deduped)} new + {len(preserved)} preserved")

    # Sort preserved by likes so they still appear high in search
    merged = sorted(
        preserved + deduped,
        key=lambda x: x.get("likes", 0),
        reverse=True,
    )

    # Quick stats
    likes_dist = [e.get("likes", 0) for e in merged]
    sdks = {}
    for e in merged:
        sdks[e.get("sdk", "unknown")] = sdks.get(e.get("sdk", "unknown"), 0) + 1

    print(f"\n=== Catalog summary ===")
    print(f"  Total entries:    {len(merged)}")
    print(f"  Max likes:        {max(likes_dist) if likes_dist else 0}")
    print(f"  Median likes:     {sorted(likes_dist)[len(likes_dist)//2] if likes_dist else 0}")
    print(f"  Min likes (kept): {min(likes_dist) if likes_dist else 0}")
    print(f"  SDK breakdown:")
    for sdk, count in sorted(sdks.items(), key=lambda x: -x[1]):
        print(f"    {sdk:<12}  {count}")

    # Top 10 preview
    print(f"\n=== Top 10 by likes ===")
    for e in merged[:10]:
        tag = ""
        if e.get("_real"):
            tag = "[REAL]"
        elif "DECOY" in (e.get("summary") or ""):
            tag = "[DECOY]"
        print(f"  {e['space_id']:<55} likes={e['likes']:<6} {tag}")

    if args.dry_run:
        print("\n[DRY-RUN] No file written.")
        return

    # Write
    CATALOG_FILE.write_text(json.dumps(merged, indent=2, default=str))
    print(f"\n✓ Wrote catalog to {CATALOG_FILE} ({len(merged)} entries)")

    # Size check
    size_kb = CATALOG_FILE.stat().st_size / 1024
    print(f"  File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
