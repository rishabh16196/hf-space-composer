"""
Generate synthetic fixtures (cards + mock responses) for all Spaces in the catalog.

Why: hand-crafting 14+ cards and N tasks × M Spaces × inputs of mock responses
is tedious and error-prone. This script generates plausible synthetic cards
and stub responses, keeping the env runnable end-to-end without requiring
real HF API recordings.

The synthetic responses are domain-aware (audio Spaces return transcript-like
strings, vision Spaces return caption-like strings, etc.) but deterministic
based on inputs.

Run:
    python scripts/generate_fixtures.py
"""

import hashlib
import json
import sys
from pathlib import Path

# Allow running from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = ROOT / "fixtures"
CATALOG_FILE = FIXTURES_DIR / "space_catalog.json"
TASKS_FILE = FIXTURES_DIR / "tasks.json"
CARDS_DIR = FIXTURES_DIR / "cards"
RESPONSES_DIR = FIXTURES_DIR / "responses"


# ---------------------------------------------------------------------------
# Card templates by tag
# ---------------------------------------------------------------------------

def make_card(space_entry):
    """Generate a plausible card for a Space."""
    space_id = space_entry["space_id"]
    tags = set(space_entry.get("tags", []))
    desc = space_entry.get("summary", "")

    if {"audio", "transcription"} & tags or {"asr"} & tags:
        return {
            "space_id": space_id,
            "description": desc + " Use the 'language' parameter for better accuracy.",
            "input_schema": {
                "audio_url": {"type": "string", "description": "URL to audio file (wav/mp3)", "required": True},
                "language": {"type": "string", "description": "ISO 639-1 code (optional)", "required": False},
            },
            "output_schema": {
                "transcript": {"type": "string", "description": "Full transcription"},
                "language_detected": {"type": "string", "description": "Detected language"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"translation"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "text": {"type": "string", "description": "Text to translate", "required": True},
                "src": {"type": "string", "description": "Source language code", "required": True},
                "tgt": {"type": "string", "description": "Target language code", "required": True},
            },
            "output_schema": {
                "text": {"type": "string", "description": "Translated text"},
            },
            "license": "cc-by-nc-4.0",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"summarization"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "text": {"type": "string", "description": "Text to summarize (max ~1024 tokens)", "required": True},
                "max_length": {"type": "integer", "description": "Max summary length", "required": False},
            },
            "output_schema": {
                "summary": {"type": "string", "description": "Generated summary"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"vision", "captioning"} & tags or "captioning" in tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "image_url": {"type": "string", "description": "URL to image", "required": True},
            },
            "output_schema": {
                "caption": {"type": "string", "description": "Image caption"},
            },
            "license": "bsd-3-clause",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"ocr", "text-extraction"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "image_url": {"type": "string", "description": "URL to image with text", "required": True},
            },
            "output_schema": {
                "extracted_text": {"type": "string", "description": "OCR-extracted text"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"sentiment"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "text": {"type": "string", "description": "Text to classify", "required": True},
            },
            "output_schema": {
                "label": {"type": "string", "description": "positive | negative | neutral"},
                "score": {"type": "number", "description": "Confidence 0-1"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"diarization"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "audio_url": {"type": "string", "description": "URL to multi-speaker audio", "required": True},
            },
            "output_schema": {
                "segments": {"type": "list", "description": "[{speaker, start, end, text}]"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"pdf"} & tags or {"document"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "pdf_url": {"type": "string", "description": "URL to PDF document", "required": True},
            },
            "output_schema": {
                "extracted_text": {"type": "string", "description": "Plain text from PDF"},
            },
            "license": "apache-2.0",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"ner", "entity-extraction"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "text": {"type": "string", "description": "Text to extract entities from", "required": True},
            },
            "output_schema": {
                "persons": {"type": "list", "description": "List of person names"},
                "organizations": {"type": "list", "description": "List of org names"},
                "locations": {"type": "list", "description": "List of location names"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"code"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "code_snippet": {"type": "string", "description": "Code to explain", "required": True},
                "language": {"type": "string", "description": "Programming language", "required": False},
            },
            "output_schema": {
                "explanation": {"type": "string", "description": "Natural-language explanation"},
            },
            "license": "apache-2.0",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    elif {"tts", "speech-synthesis"} & tags:
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "text": {"type": "string", "description": "Text to synthesize", "required": True},
                "language": {"type": "string", "description": "Target language code", "required": False},
            },
            "output_schema": {
                "audio_url": {"type": "string", "description": "URL to generated audio file"},
            },
            "license": "mpl-2.0",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }
    else:
        # Generic catch-all
        return {
            "space_id": space_id,
            "description": desc,
            "input_schema": {
                "input": {"type": "string", "description": "Generic input", "required": True},
            },
            "output_schema": {
                "output": {"type": "string", "description": "Generic output"},
            },
            "license": "mit",
            "sdk": space_entry.get("sdk", "gradio"),
            "tags": list(tags),
        }


# ---------------------------------------------------------------------------
# Synthetic response generation
# ---------------------------------------------------------------------------

DECOY_SPACES = {"fake-org/whisper-clone-v2", "spammer/audio-to-text-fast", "fake-org/blip-clone"}


def make_response(space_entry, inputs):
    """Generate a deterministic synthetic response for a Space + inputs."""
    space_id = space_entry["space_id"]
    tags = set(space_entry.get("tags", []))

    # Decoys return junk
    if space_id in DECOY_SPACES:
        return {
            "success": True,  # Decoy SUCCEEDS but produces fabricated output (Auditor catches)
            "output": {"transcript": "lorem ipsum dolor sit amet HALLUCINATED garbage random text"},
            "error": None,
        }

    # Audio transcription
    if {"asr", "transcription"} & tags:
        lang = inputs.get("language", "en")
        return {
            "success": True,
            "output": {
                "transcript": (
                    f"This is a synthetic transcript of the audio at {inputs.get('audio_url', 'unknown')}. "
                    f"It contains roughly five sentences of plausible content for testing purposes. "
                    f"The detected language was {lang}. The audio appears to be a news report covering a recent event. "
                    f"Multiple speakers may have been present. The recording lasted approximately two minutes."
                ),
                "language_detected": lang,
            },
            "error": None,
        }

    # Translation
    if {"translation"} & tags:
        text = inputs.get("text", "")
        tgt = inputs.get("tgt", "en")
        return {
            "success": True,
            "output": {
                "text": f"[translated to {tgt}] {text[:200]}",
            },
            "error": None,
        }

    # Summarization
    if {"summarization"} & tags:
        text = inputs.get("text", "")
        max_len = inputs.get("max_length", 100)
        return {
            "success": True,
            "output": {
                "summary": f"This is a synthetic news report summary covering the main story in approximately {max_len} words. {text[:150]}..."
            },
            "error": None,
        }

    # Image captioning
    if "captioning" in tags or {"vision"} & tags:
        return {
            "success": True,
            "output": {
                "caption": f"A photograph showing the scene depicted at {inputs.get('image_url', 'unknown')}. The image contains people and an outdoor setting.",
            },
            "error": None,
        }

    # OCR
    if {"ocr"} & tags:
        return {
            "success": True,
            "output": {
                "extracted_text": "Synthetic OCR output: TODAY'S MENU - Coffee $3.50 - Sandwich $8 - Salad $7",
            },
            "error": None,
        }

    # Sentiment
    if {"sentiment"} & tags:
        text = inputs.get("text", "")
        # Crude rule: presence of positive words
        positive_words = ["good", "great", "happy", "excellent", "love"]
        neg_words = ["bad", "terrible", "hate", "awful", "sad"]
        if any(w in text.lower() for w in positive_words):
            label = "positive"
        elif any(w in text.lower() for w in neg_words):
            label = "negative"
        else:
            label = "neutral"
        return {
            "success": True,
            "output": {"label": label, "score": 0.85},
            "error": None,
        }

    # Diarization
    if {"diarization"} & tags:
        return {
            "success": True,
            "output": {
                "segments": [
                    {"speaker": "speaker_1", "start": 0.0, "end": 30.0, "text": "Synthetic speaker 1 content."},
                    {"speaker": "speaker_2", "start": 30.5, "end": 60.0, "text": "Synthetic speaker 2 content."},
                ],
            },
            "error": None,
        }

    # PDF extraction
    if {"pdf"} & tags or {"document"} & tags and "extraction" in tags:
        return {
            "success": True,
            "output": {
                "extracted_text": "Synthetic PDF extraction. " * 50,
            },
            "error": None,
        }

    # NER
    if {"ner", "entity-extraction"} & tags:
        text = inputs.get("text", "")
        return {
            "success": True,
            "output": {
                "persons": ["Jane Doe"] if "Jane Doe" in text else ["Person A"],
                "organizations": ["Acme Corp"] if "Acme" in text else ["Org X"],
                "locations": [w for w in ["Tokyo", "Berlin", "Paris", "London"] if w in text] or ["LocationY"],
            },
            "error": None,
        }

    # Code explanation
    if {"code"} & tags:
        snippet = inputs.get("code_snippet", "")
        return {
            "success": True,
            "output": {
                "explanation": (
                    "This Python function implements a recursive fibonacci sequence "
                    "computation. It returns n directly for n < 2, otherwise it recursively "
                    "computes fib(n-1) + fib(n-2)."
                    if "fib" in snippet else
                    f"Synthetic explanation of code: {snippet[:100]}"
                ),
            },
            "error": None,
        }

    # TTS
    if {"tts"} & tags:
        return {
            "success": True,
            "output": {
                "audio_url": "https://example.com/generated/tts_synthetic.wav",
            },
            "error": None,
        }

    # Default
    return {
        "success": True,
        "output": {"output": f"synthetic output for {space_id}"},
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def main():
    catalog = json.loads(CATALOG_FILE.read_text())
    tasks = json.loads(TASKS_FILE.read_text()) if TASKS_FILE.exists() else []

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate cards for ALL Spaces in catalog
    cards_made = 0
    for space in catalog:
        space_id = space["space_id"]
        safe = space_id.replace("/", "_")
        card_path = CARDS_DIR / f"{safe}.json"
        if card_path.exists():
            continue  # Don't overwrite hand-crafted cards
        card = make_card(space)
        card_path.write_text(json.dumps(card, indent=2))
        cards_made += 1

    # Index for quick lookup
    catalog_by_id = {s["space_id"]: s for s in catalog}

    # Generate mock responses based on each task's gold_pipeline (if any)
    responses_made = 0
    for task in tasks:
        task_id = task["task_id"]
        task_responses_dir = RESPONSES_DIR / task_id
        task_responses_dir.mkdir(parents=True, exist_ok=True)

        # Resolve <input.X> and <stepN.Y> placeholders by substituting plausible values
        gold = task.get("gold_pipeline", [])
        # Build a "resolved" version of each step's inputs
        resolved_steps = []
        for step in gold:
            inputs = {}
            for k, v in step.get("inputs", {}).items():
                if isinstance(v, str) and v.startswith("<input."):
                    field = v.replace("<input.", "").replace(">", "")
                    inputs[k] = task.get("input", {}).get(field, "synthetic_value")
                elif isinstance(v, str) and v.startswith("<step"):
                    inputs[k] = "synthetic_value_from_previous_step"
                else:
                    inputs[k] = v
            resolved_steps.append({"space_id": step["space_id"], "inputs": inputs})

        for step in resolved_steps:
            space_id = step["space_id"]
            inputs = step["inputs"]
            space_entry = catalog_by_id.get(space_id)
            if not space_entry:
                continue
            response = make_response(space_entry, inputs)

            # Hash inputs for filename
            input_hash = hashlib.sha1(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            safe = space_id.replace("/", "_")
            response_path = task_responses_dir / f"{safe}__{input_hash}.json"
            response_path.write_text(json.dumps(response, indent=2))
            responses_made += 1

        # Also generate fallback responses for each Space in catalog with the
        # task's "input" values, to support agents that try various inputs
        for space in catalog:
            if space["space_id"] in DECOY_SPACES:
                # For decoys, generate response with empty inputs hash so any call hits it
                safe = space["space_id"].replace("/", "_")
                # Use a wildcard hash by serializing common input shapes
                for sample_inputs in [
                    {"audio_url": task.get("input", {}).get("audio_url", "")},
                    {"text": "any text"},
                    {"image_url": task.get("input", {}).get("image_url", "")},
                ]:
                    if any(sample_inputs.values()):
                        response = make_response(space, sample_inputs)
                        h = hashlib.sha1(
                            json.dumps(sample_inputs, sort_keys=True, default=str).encode()
                        ).hexdigest()[:12]
                        path = task_responses_dir / f"{safe}__{h}.json"
                        path.write_text(json.dumps(response, indent=2))
                        responses_made += 1

    print(f"✓ Generated {cards_made} cards in {CARDS_DIR}")
    print(f"✓ Generated {responses_made} mock responses in {RESPONSES_DIR}")
    print(f"✓ Catalog has {len(catalog)} Spaces, {len(tasks)} tasks")


if __name__ == "__main__":
    main()
