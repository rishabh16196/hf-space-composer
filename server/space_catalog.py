"""
Space Catalog: unified interface for searching and reading HuggingFace Spaces.

Supports four modes (via SPACES_MODE env var):
  - "mock":   Return deterministic results from fixtures/space_catalog.json
              and fixtures/cards/*.json. Fast (~10ms). Used for training.
  - "live":   Query the real HuggingFace Hub API via huggingface_hub.
              Slow (1-3s per call). Used for demos.
  - "record": Live calls + cache to fixtures (for refreshing fixtures).
  - "hybrid": Try cache first, fall back to live on miss, auto-cache response.
              Best of both worlds: fast for known Spaces, self-extending for
              new ones. Ideal for iterative development and demo safety net.

This module isolates Hub API access so the rest of the env is mode-agnostic.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

MODE_MOCK = "mock"
MODE_LIVE = "live"
MODE_RECORD = "record"
MODE_HYBRID = "hybrid"

ALL_MODES = {MODE_MOCK, MODE_LIVE, MODE_RECORD, MODE_HYBRID}


def get_mode() -> str:
    """Return active catalog mode."""
    return os.getenv("SPACES_MODE", MODE_MOCK).lower()


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

# Locate fixtures dir relative to this file
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CATALOG_FIXTURE = FIXTURES_DIR / "space_catalog.json"
CARDS_DIR = FIXTURES_DIR / "cards"


def _load_catalog_fixture() -> List[Dict[str, Any]]:
    """Load the mock catalog index. Returns [] if fixture missing."""
    if not CATALOG_FIXTURE.exists():
        return []
    with open(CATALOG_FIXTURE) as f:
        return json.load(f)


def _load_card_fixture(space_id: str) -> Optional[Dict[str, Any]]:
    """Load a Space card from fixtures. Returns None if not cached."""
    safe_id = space_id.replace("/", "_")
    card_path = CARDS_DIR / f"{safe_id}.json"
    if not card_path.exists():
        return None
    with open(card_path) as f:
        return json.load(f)


def _save_card_fixture(space_id: str, card: Dict[str, Any]) -> None:
    """Write a card to fixtures (used in record/hybrid modes)."""
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = space_id.replace("/", "_")
    card_path = CARDS_DIR / f"{safe_id}.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)


def _append_to_catalog_fixture(entry: Dict[str, Any]) -> None:
    """Append a Space entry to the catalog fixture if not already present.

    Used in hybrid mode when live search surfaces a Space we've never seen.
    Keeps the catalog growing organically as agents explore.
    """
    try:
        catalog = _load_catalog_fixture()
        existing_ids = {s.get("space_id") for s in catalog}
        if entry.get("space_id") in existing_ids:
            return
        catalog.append(entry)
        # Re-sort by likes desc to preserve order
        catalog.sort(key=lambda x: x.get("likes", 0) or 0, reverse=True)
        with open(CATALOG_FIXTURE, "w") as f:
            json.dump(catalog, f, indent=2, default=str)
    except Exception:
        # Best-effort; don't fail an action if we can't write
        pass


# ---------------------------------------------------------------------------
# Live HF API access
# ---------------------------------------------------------------------------

def _live_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search live HF Hub for Spaces matching query."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    api = HfApi()
    spaces = api.list_spaces(search=query, limit=top_k, full=True)

    results = []
    for s in spaces:
        results.append({
            "space_id": s.id,
            "name": s.id.split("/")[-1],
            "author": s.author or s.id.split("/")[0],
            "downloads": getattr(s, "likes", 0),  # likes used as proxy
            "likes": getattr(s, "likes", 0),
            "tags": list(getattr(s, "tags", []) or []),
            "sdk": getattr(s, "sdk", "unknown"),
            "summary": (getattr(s, "description", "") or "")[:200],
            "last_modified": str(getattr(s, "last_modified", "")),
        })
    return results


def _live_read_card(space_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full card from HF Hub for given Space."""
    try:
        from huggingface_hub import HfApi, space_info
    except ImportError:
        return None

    try:
        info = space_info(space_id, full=True)
    except Exception:
        return None

    card_data = getattr(info, "cardData", {}) or {}
    return {
        "space_id": space_id,
        "description": (getattr(info, "description", "") or "")[:1000],
        "input_schema": card_data.get("input_schema", {}),
        "output_schema": card_data.get("output_schema", {}),
        "example_inputs": card_data.get("example_inputs", []),
        "license": card_data.get("license", "unknown"),
        "hardware": getattr(info, "hardware", "cpu"),
        "sdk": getattr(info, "sdk", "unknown"),
        "tags": list(getattr(info, "tags", []) or []),
        "last_modified": str(getattr(info, "last_modified", "")),
    }


# ---------------------------------------------------------------------------
# Public Catalog API
# ---------------------------------------------------------------------------

class SpaceCatalog:
    """Unified Space catalog interface (mock | live | record | hybrid)."""

    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = (mode or get_mode()).lower()
        if self.mode not in ALL_MODES:
            raise ValueError(f"Unknown SPACES_MODE: {self.mode}. Valid: {ALL_MODES}")
        # Load mock catalog for any mode that consults it
        # Live mode doesn't need it, but hybrid does (as primary source)
        if self.mode == MODE_LIVE:
            self._mock_catalog = []
        else:
            self._mock_catalog = _load_catalog_fixture()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Spaces by keyword query. Returns ordered list of results."""
        if self.mode == MODE_MOCK:
            return self._mock_search(query, top_k)
        elif self.mode == MODE_LIVE:
            return _live_search(query, top_k)
        elif self.mode == MODE_RECORD:
            return _live_search(query, top_k)
        elif self.mode == MODE_HYBRID:
            # Try local catalog first
            local = self._mock_search(query, top_k)
            if len(local) >= top_k:
                return local
            # Supplement with live results for anything we didn't find locally
            live = _live_search(query, top_k * 2)
            # Cache newly-discovered entries back to the catalog
            local_ids = {e.get("space_id") for e in local}
            for entry in live:
                if entry.get("space_id") and entry["space_id"] not in local_ids:
                    _append_to_catalog_fixture(entry)
                    local.append(entry)
                    local_ids.add(entry["space_id"])
                    if len(local) >= top_k:
                        break
            # Re-load catalog for subsequent searches
            self._mock_catalog = _load_catalog_fixture()
            return local[:top_k]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def read_card(self, space_id: str) -> Optional[Dict[str, Any]]:
        """Read full card for a Space."""
        if self.mode == MODE_MOCK:
            return _load_card_fixture(space_id)
        elif self.mode == MODE_LIVE:
            return _live_read_card(space_id)
        elif self.mode == MODE_RECORD:
            card = _live_read_card(space_id)
            if card is not None:
                _save_card_fixture(space_id, card)
            return card
        elif self.mode == MODE_HYBRID:
            # Try cache first
            card = _load_card_fixture(space_id)
            if card is not None:
                return card
            # Miss: fetch live and cache
            card = _live_read_card(space_id)
            if card is not None:
                card["_hybrid_fetched"] = True
                _save_card_fixture(space_id, card)
            return card
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _mock_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Mock search: keyword-match filter + likes as tiebreaker.

        Scoring logic:
          - keyword_score = number of query terms that appear in
            name / summary / tags / space_id
          - ONLY entries with keyword_score > 0 qualify
          - final_score = keyword_score * 10 + log(likes+1)
          - (keyword match dominates; likes is tiebreaker among matches)
        """
        import math
        if not self._mock_catalog:
            return []
        q_terms = [t for t in query.lower().split() if len(t) >= 2]
        if not q_terms:
            return []

        scored: List[tuple] = []
        for entry in self._mock_catalog:
            haystack_parts = [
                entry.get("name", "").lower(),
                (entry.get("summary", "") or "").lower(),
                " ".join(entry.get("tags", []) or []).lower(),
                entry.get("space_id", "").lower(),
            ]
            haystack = " ".join(haystack_parts)

            keyword_score = sum(1 for t in q_terms if t in haystack)
            if keyword_score == 0:
                continue

            likes = max(0, entry.get("likes", 0) or 0)
            final_score = keyword_score * 10.0 + math.log1p(likes)
            scored.append((final_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


# Convenience module-level singleton (lazy-init)
_default_catalog: Optional[SpaceCatalog] = None


def get_catalog() -> SpaceCatalog:
    """Return process-wide default catalog."""
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = SpaceCatalog()
    return _default_catalog


def reset_catalog() -> None:
    """Force re-initialization of the default catalog (e.g., after mode change)."""
    global _default_catalog
    _default_catalog = None
