"""
Replace mock/model Space IDs in task gold_pipelines with REAL verified Spaces.

Many of our gold_pipelines reference IDs like `openai/whisper-large-v3` (a model,
not a Space) or `facebook/nllb-200` (also a model). These need to be replaced
with actual Space IDs that agents can call.

This script:
  1. Scans all task gold_pipelines for Space IDs
  2. For each ID that isn't API-verified (or doesn't exist as a Space), finds
     the best real verified equivalent from the catalog
  3. Applies the replacement (both space_id and input field names where needed)
  4. Writes the updated tasks.json

Matching heuristic:
  - Prefer Spaces with _api_verified == True AND measured_inference_s
  - Match by domain tag overlap (≥2 common tags)
  - Prefer high-likes Spaces
  - Preserve input field names from the real Space's card

Usage:
    python scripts/upgrade_gold_pipelines.py           # preview
    python scripts/upgrade_gold_pipelines.py --apply   # write changes
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASKS_FILE = ROOT / "fixtures" / "tasks.json"
CARDS_DIR = ROOT / "fixtures" / "cards"
CATALOG_FILE = ROOT / "fixtures" / "space_catalog.json"


# Hand-curated hints: for each mock Space ID in our gold_pipelines, which
# real Space should replace it. These are chosen because they're API-verified
# and known to work.
PREFERRED_MAPPING: Dict[str, str] = {
    # Verified replacements — chosen because both Spaces work and APIs checked
    "openai/whisper-large-v3":              "hf-audio/whisper-large-v3",
    "facebook/nllb-200":                    "UNESCO/nllb",
    "facebook/nllb-200-v2":                 "UNESCO/nllb",  # deprecation drift target
    "microsoft/trocr-base":                 "merterbak/DeepSeek-OCR-Demo",
    "bigcode/code-explainer":               "Qwen/Qwen2.5-Coder-Artifacts",
    "coqui/xtts-v2":                        "innoai/Edge-TTS-Text-to-Speech",
    "facebook/bart-large-cnn":              "pszemraj/summarize-long-text",
    "Salesforce/blip-image-captioning":     "fancyfeast/joy-caption-pre-alpha",

    # Newly found verified replacements for the "long-tail" domains
    "cardiffnlp/twitter-roberta-sentiment": "Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis",
    "dslim/bert-ner":                       "fastino/gliner2-official-demo",
    "pdf-tools/pdf-extractor":              "xiaoyao9184/marker",
    "pyannote/speaker-diarization":         "wenet-e2e/wespeaker_demo",
}

# Input field mappings — when we swap the Space, some input keys rename too.
# Keyed by (old_space_id, new_space_id).
INPUT_KEY_MAPPING: Dict[Tuple[str, str], Dict[str, str]] = {
    ("openai/whisper-large-v3", "hf-audio/whisper-large-v3"): {
        "audio_url": "inputs",
        "language": "task",   # lose language, just use "transcribe"
    },
    ("facebook/nllb-200", "UNESCO/nllb"): {
        "text": "text",
        "src": "src_lang",
        "tgt": "tgt_lang",
    },
    ("facebook/nllb-200-v2", "UNESCO/nllb"): {
        "text": "text",
        "src": "src_lang",
        "tgt": "tgt_lang",
    },
    ("bigcode/code-explainer", "Qwen/Qwen2.5-Coder-Artifacts"): {
        "code_snippet": "query",
    },
    ("coqui/xtts-v2", "innoai/Edge-TTS-Text-to-Speech"): {
        "text": "text",
        "language": "voice",   # lose simple language, need proper voice id
    },
    ("microsoft/trocr-base", "merterbak/DeepSeek-OCR-Demo"): {
        "image_url": "image",
    },
    ("cardiffnlp/twitter-roberta-sentiment", "Gradio-Blocks/Multilingual-Aspect-Based-Sentiment-Analysis"): {
        "text": "text",
    },
    ("dslim/bert-ner", "fastino/gliner2-official-demo"): {
        "text": "text",
    },
    ("pdf-tools/pdf-extractor", "xiaoyao9184/marker"): {
        "pdf_url": "filename",
    },
    ("pyannote/speaker-diarization", "wenet-e2e/wespeaker_demo"): {
        "audio_url": "Speaker#1",
    },
}

# Language code → NLLB language name mapping
LANG_CODE_TO_NAME = {
    "en": "English", "fr": "French", "es": "Spanish", "hi": "Hindi",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "zh": "Chinese (Simplified)", "ja": "Japanese", "ko": "Korean", "ar": "Modern Standard Arabic",
}


def load_card(space_id: str) -> Optional[Dict[str, Any]]:
    p = CARDS_DIR / (space_id.replace("/", "_") + ".json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_verified_equivalent(
    target_tags: List[str],
    catalog: List[Dict[str, Any]],
    primary_domain_tag: Optional[str] = None,
) -> Optional[str]:
    """Find best API-verified Space by DOMAIN match + likes.

    Requires:
      - primary_domain_tag (if given) MUST appear in candidate's tags
      - OR at least 2 of the target_tags appear in candidate
      - Never match if the candidate has conflicting primary domain tags
        (e.g., "ner" target should never match a Space tagged only "tts")
    """
    best = None
    best_score = -1
    target = set(t.lower() for t in target_tags)
    primary = (primary_domain_tag or "").lower()
    # Tags that disqualify a candidate if it's primary-tagged with them
    # but we need a different primary. Prevents sentiment → tts matches.
    conflicting_primaries = {
        "sentiment": {"tts", "speech-synthesis", "translation", "ner", "ocr", "diffusion",
                      "image-generation", "text-to-image"},
        "ner": {"tts", "speech-synthesis", "translation", "summarization", "diffusion",
                "image-generation", "text-to-image", "sentiment"},
        "summarization": {"tts", "speech-synthesis", "translation", "ner", "ocr",
                          "diffusion", "image-generation"},
        "diarization": {"tts", "translation", "text-to-image", "image-generation",
                        "diffusion", "summarization"},
        "pdf": {"tts", "audio", "asr", "text-to-image"},
        "ocr": {"tts", "audio", "asr", "translation", "text-to-image"},
        "captioning": {"tts", "audio", "asr", "translation", "summarization"},
    }
    conflicts = conflicting_primaries.get(primary, set())

    for entry in catalog:
        sid = entry.get("space_id")
        if not sid:
            continue
        card = load_card(sid)
        if not card:
            continue
        if not card.get("_api_verified"):
            continue
        tags = set(t.lower() for t in (card.get("tags") or []))

        # Required: primary_domain_tag must appear
        if primary and primary not in tags:
            continue
        # Rejection: must not have conflicting primary markers
        if conflicts and (tags & conflicts):
            continue
        # Scoring: overlap across target + likes
        overlap = len(tags & target)
        if overlap < 1:
            continue
        likes = entry.get("likes", 0) or 0
        score = overlap * 100 + (likes / 100.0)
        if score > best_score:
            best_score = score
            best = sid
    return best


def find_replacement(mock_id: str, catalog: List[Dict[str, Any]]) -> Optional[str]:
    """Pick a real Space ID to replace this mock one.

    Whitelist-only policy: only replace when we have a hand-curated mapping
    to a real verified Space. Refuse to guess — better to keep a mock ID than
    pick a wrong domain.
    """
    # 1. Is the mock_id itself actually a verified Space?
    mock_card = load_card(mock_id)
    if mock_card and mock_card.get("_api_verified"):
        return mock_id

    # 2. Check curated mapping
    if mock_id in PREFERRED_MAPPING:
        preferred = PREFERRED_MAPPING[mock_id]
        if preferred:
            card = load_card(preferred)
            if card and card.get("_api_verified"):
                return preferred

    # 3. No safe replacement — leave as-is (gold_pipeline keeps mock,
    #    env will use synthetic response fixtures for it)
    return None


def remap_inputs(
    old_space_id: str,
    new_space_id: str,
    old_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate gold_pipeline inputs when swapping Spaces."""
    mapping = INPUT_KEY_MAPPING.get((old_space_id, new_space_id))
    if not mapping:
        # No explicit mapping — best-effort: use real card's input schema
        card = load_card(new_space_id)
        if not card:
            return old_inputs
        schema = card.get("input_schema", {}) or {}
        new_inputs = {}
        for field_name, field_spec in schema.items():
            # Try to find a matching old input
            if field_name in old_inputs:
                new_inputs[field_name] = old_inputs[field_name]
            elif len(old_inputs) == 1 and isinstance(field_spec, dict) and field_spec.get("required"):
                # Single-input case: map the lone value
                new_inputs[field_name] = next(iter(old_inputs.values()))
        return new_inputs if new_inputs else old_inputs

    new_inputs = {}
    for old_k, new_k in mapping.items():
        if old_k not in old_inputs:
            continue
        val = old_inputs[old_k]
        # Special handling for whisper: old had (audio_url, language), new has (inputs, task)
        if (old_space_id == "openai/whisper-large-v3"
            and new_space_id == "hf-audio/whisper-large-v3"):
            if old_k == "audio_url":
                new_inputs["inputs"] = val
            elif old_k == "language":
                new_inputs["task"] = "transcribe"
            continue
        # Special handling for NLLB: src/tgt short codes → language names
        if (old_space_id in ("facebook/nllb-200", "facebook/nllb-200-v2")
            and new_space_id == "UNESCO/nllb"):
            if old_k in ("src", "tgt") and isinstance(val, str) and len(val) <= 5:
                val = LANG_CODE_TO_NAME.get(val.lower(), val)
        # Default rename
        new_inputs[new_k] = val

    # If we dropped 'task' for whisper, default to transcribe
    if (old_space_id == "openai/whisper-large-v3"
        and new_space_id == "hf-audio/whisper-large-v3"
        and "task" not in new_inputs):
        new_inputs["task"] = "transcribe"

    # Preserve any input keys that didn't have a mapping
    for k, v in old_inputs.items():
        if k not in mapping and k not in new_inputs.values():
            new_inputs.setdefault(k, v)

    return new_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes (default: preview)")
    args = parser.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())
    catalog = json.loads(CATALOG_FILE.read_text())

    changed_tasks = 0
    total_steps = 0
    replaced_steps = 0
    replacements: Dict[str, str] = {}

    for task in tasks:
        gold = task.get("gold_pipeline", [])
        if not gold:
            continue
        for step in gold:
            total_steps += 1
            old = step.get("space_id")
            if not old:
                continue
            # Check if old is already verified real
            old_card = load_card(old)
            if old_card and old_card.get("_api_verified"):
                continue

            new = find_replacement(old, catalog)
            if not new or new == old:
                continue

            # Apply replacement
            replacements[old] = new
            step["space_id"] = new
            step["inputs"] = remap_inputs(old, new, step.get("inputs", {}))
            replaced_steps += 1
        changed_tasks += 1 if any(step.get("space_id") in replacements.values() for step in gold) else 0

    print(f"Total gold_pipeline steps: {total_steps}")
    print(f"Replaced steps:            {replaced_steps}")
    print(f"Unique substitutions:      {len(replacements)}")
    print()
    for old, new in sorted(replacements.items()):
        print(f"  {old}  →  {new}")

    if args.apply:
        TASKS_FILE.write_text(json.dumps(tasks, indent=2, default=str))
        print(f"\n✓ Wrote updated tasks to {TASKS_FILE}")
    else:
        print(f"\n[preview] Pass --apply to write changes")


if __name__ == "__main__":
    main()
