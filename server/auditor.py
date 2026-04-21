"""
Auditor (Fleet AI sub-theme).

Rule-based oversight agent that monitors agent actions and Space outputs.
Raises flags for various failure modes:
  - INFO     — informational, no penalty (e.g., agent skipped reading card)
  - WARNING  — small penalty (e.g., calling Space without reading card first)
  - ERROR    — significant penalty (e.g., output schema mismatch)
  - CRITICAL — heavy penalty (e.g., calling deprecated Space, safety violation)

Day 3: pure rule-based. Optional LLM-judge augmentation later (cached).
"""

from typing import Any, Dict, List, Optional


# Decoy / known-bad Spaces that should never be used
DECOY_SPACES = {
    "fake-org/whisper-clone-v2",
    "spammer/audio-to-text-fast",
    "fake-org/blip-clone",
}


class Auditor:
    """Stateful auditor — accumulates flags across an episode."""

    def __init__(self) -> None:
        self.flags: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.flags = []

    def review_action(
        self,
        step: int,
        action_type: str,
        payload: Dict[str, Any],
        success: bool,
        error: Optional[str],
        cards_read: Dict[str, Dict[str, Any]],
        space_call_history: List[Dict[str, Any]],
        catalog: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Review a single action. Returns list of new flags.

        Args:
            catalog: optional SpaceCatalog for faster-alternative detection
        """
        new_flags: List[Dict[str, Any]] = []

        if action_type == "call_space":
            new_flags.extend(
                self._audit_call_space(step, payload, success, cards_read, space_call_history, catalog)
            )
        elif action_type == "search_spaces":
            new_flags.extend(self._audit_search(step, payload))
        elif action_type == "submit":
            # Submission audits handled in `review_submission`
            pass

        # Apply error-based flags universally
        if not success and error:
            new_flags.append(self._flag(
                step=step,
                severity="warning",
                message=f"Action {action_type} failed: {error[:120]}",
                action_type=action_type,
            ))

        self.flags.extend(new_flags)
        return new_flags

    def review_submission(
        self,
        step: int,
        submitted: Optional[Dict[str, Any]],
        expected_schema: Dict[str, Any],
        space_call_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Audit final submission. Returns list of new flags."""
        new_flags: List[Dict[str, Any]] = []

        if submitted is None or not isinstance(submitted, dict):
            new_flags.append(self._flag(
                step=step,
                severity="critical",
                message="Submission missing or not a dict",
                action_type="submit",
            ))
            self.flags.extend(new_flags)
            return new_flags

        # Field presence check
        if expected_schema:
            for field in expected_schema:
                if field not in submitted:
                    new_flags.append(self._flag(
                        step=step,
                        severity="error",
                        message=f"Submitted answer missing expected field '{field}'",
                        action_type="submit",
                    ))
                elif submitted[field] in (None, "", [], {}):
                    new_flags.append(self._flag(
                        step=step,
                        severity="warning",
                        message=f"Submitted answer has empty value for '{field}'",
                        action_type="submit",
                    ))

        # Hallucination heuristic: if no successful Space calls but content submitted
        successful_calls = [c for c in space_call_history if c.get("success")]
        if not successful_calls and submitted:
            new_flags.append(self._flag(
                step=step,
                severity="error",
                message="Submitted content without any successful Space calls",
                action_type="submit",
            ))

        self.flags.extend(new_flags)
        return new_flags

    # ------------------------------------------------------------------
    # Internal audits
    # ------------------------------------------------------------------

    def _audit_call_space(
        self,
        step: int,
        payload: Dict[str, Any],
        success: bool,
        cards_read: Dict[str, Dict[str, Any]],
        space_call_history: List[Dict[str, Any]],
        catalog: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        space_id = payload.get("space_id", "")

        # Decoy / known-bad Space
        if space_id in DECOY_SPACES:
            flags.append(self._flag(
                step=step,
                severity="critical",
                message=f"Called known-bad/decoy Space '{space_id}' — fabricated outputs",
                action_type="call_space",
                space_id=space_id,
            ))

        # Called without reading card
        if space_id and space_id not in cards_read:
            flags.append(self._flag(
                step=step,
                severity="warning",
                message=f"Called '{space_id}' without reading its card first",
                action_type="call_space",
                space_id=space_id,
            ))

        # Redundant call (same space + same inputs as previous successful call)
        inputs = payload.get("inputs", {})
        for prev in space_call_history:
            if (
                prev.get("space_id") == space_id
                and prev.get("inputs") == inputs
                and prev.get("success")
            ):
                flags.append(self._flag(
                    step=step,
                    severity="warning",
                    message=f"Redundant call to '{space_id}' with identical inputs",
                    action_type="call_space",
                    space_id=space_id,
                ))
                break

        # Efficiency: slow Space used when a faster equivalent exists.
        # We ONLY fire this flag when BOTH the current Space and the alternative
        # have MEASURED latencies — never estimated — so it's always grounded
        # in real wall-clock data.
        if space_id and catalog is not None:
            card = cards_read.get(space_id)
            if card:
                # Must be measured — don't flag based on estimates
                if card.get("_latency_source") != "measured":
                    pass  # skip — we don't have real data for this one
                else:
                    this_latency = card.get("measured_latency_s") or card.get("estimated_latency_s") or 0.0
                    this_tags = set((t or "").lower() for t in card.get("tags", []) or [])
                    # Only flag if measurably slow (>20s)
                    if this_latency >= 20.0 and this_tags:
                        faster = self._find_faster_equivalent(
                            catalog, this_tags, this_latency, exclude={space_id}
                        )
                        if faster:
                            flags.append(self._flag(
                                step=step,
                                severity="warning",
                                message=(
                                    f"Used slow Space '{space_id}' (measured {this_latency}s). "
                                    f"Faster measured equivalent: '{faster['space_id']}' ({faster['latency']}s). "
                                    f"Consider speed/quality trade-off."
                                ),
                                action_type="call_space",
                                space_id=space_id,
                            ))

        return flags

    # Tags that are speed/quality descriptors, not domain classifiers.
    # These are excluded from the overlap calculation so "turbo" on a Whisper
    # Space doesn't match "turbo" on a FLUX Space.
    _SPEED_META_TAGS = {
        "fast", "slow", "medium", "very_slow", "turbo", "schnell", "flash",
        "lite", "mini", "large", "small", "base", "xl", "xxl",
    }
    # High-level domain tags — at least ONE of these must appear in the overlap
    # for the alternative to count as "same domain"
    _DOMAIN_TAGS = {
        "audio", "vision", "text", "document", "code", "multimodal",
        "translation", "asr", "transcription", "tts", "speech-synthesis",
        "captioning", "image-to-text", "text-to-image", "image-generation",
        "diffusion", "ocr", "summarization", "sentiment", "ner", "entity-extraction",
        "3d", "text-to-3d", "image-to-3d", "video", "video-generation",
        "music", "song", "song-generation", "code-generation",
    }

    def _find_faster_equivalent(
        self,
        catalog: Any,
        target_tags: set,
        current_latency: float,
        exclude: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a Space with SAME-DOMAIN tag overlap AND lower latency.

        Requirements:
          - Overlap (after removing speed-meta tags) must include ≥1 domain tag
          - Must be ≥30% faster than current
          - Non-empty meaningful tag overlap (≥2 tags)
        """
        exclude = exclude or set()
        # Strip speed-meta tags from target
        target_content = target_tags - self._SPEED_META_TAGS
        target_domain = target_content & self._DOMAIN_TAGS
        if not target_domain:
            # No identifiable domain for target — skip suggestion
            return None

        best = None
        best_latency = current_latency

        mock_catalog = getattr(catalog, "_mock_catalog", []) or []
        for entry in mock_catalog:
            sid = entry.get("space_id")
            if not sid or sid in exclude:
                continue
            entry_tags = set((t or "").lower() for t in entry.get("tags", []) or [])
            entry_content = entry_tags - self._SPEED_META_TAGS
            overlap = entry_content & target_content
            # Must share at least one DOMAIN tag AND ≥2 content tags
            if not (overlap & target_domain):
                continue
            if len(overlap) < 2:
                continue

            # Must also have a MEASURED latency — don't compare against estimates
            if entry.get("_latency_source") != "measured":
                continue
            entry_latency = entry.get("measured_latency_s") or entry.get("estimated_latency_s")
            if entry_latency is None:
                continue
            try:
                entry_latency = float(entry_latency)
            except (TypeError, ValueError):
                continue
            if entry_latency < best_latency * 0.7:
                best_latency = entry_latency
                best = {"space_id": sid, "latency": entry_latency}

        return best

    def _audit_search(
        self, step: int, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        query = payload.get("query", "")
        if isinstance(query, str) and len(query) < 3:
            flags.append(self._flag(
                step=step,
                severity="info",
                message=f"Very short search query: '{query}' — likely too unspecific",
                action_type="search_spaces",
            ))
        return flags

    def _flag(
        self,
        step: int,
        severity: str,
        message: str,
        action_type: str,
        space_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "severity": severity,
            "message": message,
            "action_type": action_type,
            "space_id": space_id or "",
        }
