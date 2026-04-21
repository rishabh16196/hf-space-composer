"""
Space Caller: invoke a Space with structured inputs.

Four modes:
  - mock:   Return cached responses from fixtures/responses/{task_id}/{space_id}_{hash}.json
  - live:   Use gradio_client.Client to invoke real Space endpoints
  - record: Live + cache responses to fixtures
  - hybrid: Try cache first (real > exact hash > any), fall back to live on miss,
            auto-cache live response with _real=true marker.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .space_catalog import (
    MODE_HYBRID,
    MODE_LIVE,
    MODE_MOCK,
    MODE_RECORD,
    get_mode,
)


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
RESPONSES_DIR = FIXTURES_DIR / "responses"


def _input_hash(inputs: Dict[str, Any]) -> str:
    """Stable hash of inputs dict for cache lookup."""
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode()).hexdigest()[:12]


def _response_path(task_id: str, space_id: str, inputs: Dict[str, Any]) -> Path:
    safe_id = space_id.replace("/", "_")
    h = _input_hash(inputs)
    return RESPONSES_DIR / task_id / f"{safe_id}__{h}.json"


def _load_cached_response(
    task_id: str, space_id: str, inputs: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Load a cached Space response. Returns None if missing."""
    path = _response_path(task_id, space_id, inputs)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_cached_response(
    task_id: str, space_id: str, inputs: Dict[str, Any], response: Dict[str, Any]
) -> None:
    """Cache a Space response to fixtures."""
    path = _response_path(task_id, space_id, inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(response, f, indent=2, default=str)


def _find_real_response_for_space(
    task_id: str, space_id: str
) -> Optional[Dict[str, Any]]:
    """Return the first response for this Space marked as real (_real: true).

    This takes priority over exact-hash matches so real recorded HF outputs
    always beat synthetic fixtures.
    """
    safe_id = space_id.replace("/", "_")
    task_dir = RESPONSES_DIR / task_id
    if not task_dir.exists():
        return None
    for path in sorted(task_dir.glob(f"{safe_id}__*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("_real"):
            return data
    return None


def _find_any_response_for_space(
    task_id: str, space_id: str
) -> Optional[Dict[str, Any]]:
    """Find any cached response for this Space in this task (input-agnostic).

    Preference order:
      1. Responses marked with "_real": true (real HF recordings)
      2. Any other cached response (synthetic)

    Used as a fallback when the exact input hash isn't cached but we have
    SOME response for this Space + task. Keeps mock training robust.
    """
    safe_id = space_id.replace("/", "_")
    task_dir = RESPONSES_DIR / task_id
    if not task_dir.exists():
        return None

    real_candidates: List[Dict[str, Any]] = []
    synthetic_candidates: List[Dict[str, Any]] = []
    for path in sorted(task_dir.glob(f"{safe_id}__*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("_real"):
            real_candidates.append(data)
        else:
            synthetic_candidates.append(data)

    if real_candidates:
        return real_candidates[0]
    if synthetic_candidates:
        return synthetic_candidates[0]
    return None


def _live_call(space_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a live HF Space via gradio_client. May raise on failure."""
    try:
        from gradio_client import Client
    except ImportError:
        return {
            "success": False,
            "error": "gradio_client not installed",
            "output": None,
        }

    try:
        client = Client(space_id)
        # Best-effort: pass kwargs and let gradio_client figure out the API
        result = client.predict(**inputs, api_name="/predict")
        return {"success": True, "output": result, "error": None}
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SpaceCaller:
    """Unified Space invocation interface."""

    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = (mode or get_mode()).lower()

    def call(
        self,
        task_id: str,
        space_id: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a Space. Returns {success, output, error}.

        Mock-mode preference order:
          1. Any cached response for this Space marked "_real": true
             (beats exact hash match to ensure real HF outputs take priority)
          2. Exact hash match on the provided inputs
          3. Any cached response for this Space in this task (synthetic)
        """
        if self.mode == MODE_MOCK:
            # 1. Prefer real recordings over anything else
            real = _find_real_response_for_space(task_id, space_id)
            if real is not None:
                return real
            # 2. Exact hash match
            cached = _load_cached_response(task_id, space_id, inputs)
            if cached is not None:
                return cached
            # 3. Any response (synthetic fallback)
            fallback = _find_any_response_for_space(task_id, space_id)
            if fallback is not None:
                return fallback
            return {
                "success": False,
                "output": None,
                "error": (
                    f"No mock response for {space_id} in task '{task_id}' "
                    f"(input_hash={_input_hash(inputs)}). "
                    f"Either Space is wrong for this task or fixture missing."
                ),
            }

        elif self.mode == MODE_LIVE:
            return _live_call(space_id, inputs)

        elif self.mode == MODE_RECORD:
            response = _live_call(space_id, inputs)
            if response.get("success"):
                # Mark as real recording so mock fallback prefers it
                response["_real"] = True
                _save_cached_response(task_id, space_id, inputs, response)
            return response

        elif self.mode == MODE_HYBRID:
            # Same preference order as mock: real > exact > any
            real = _find_real_response_for_space(task_id, space_id)
            if real is not None:
                return real
            cached = _load_cached_response(task_id, space_id, inputs)
            if cached is not None:
                return cached
            fallback = _find_any_response_for_space(task_id, space_id)
            if fallback is not None:
                return fallback
            # Cache miss: go live, mark as real, cache for next time
            response = _live_call(space_id, inputs)
            if response.get("success"):
                response["_real"] = True
                response["_hybrid_fetched"] = True
                _save_cached_response(task_id, space_id, inputs, response)
            return response

        else:
            return {
                "success": False,
                "output": None,
                "error": f"Unknown mode: {self.mode}",
            }


_default_caller: Optional[SpaceCaller] = None


def get_caller() -> SpaceCaller:
    global _default_caller
    if _default_caller is None:
        _default_caller = SpaceCaller()
    return _default_caller


def reset_caller() -> None:
    global _default_caller
    _default_caller = None
