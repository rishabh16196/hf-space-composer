"""
Enrich all cards in fixtures/cards/ with an `estimated_latency_s` field.

Strategy (heuristic, since HF API doesn't expose runtime):
  1. Start from a default by tag/category:
      - translation / summarization / sentiment / NER  → 2-4 s
      - ASR (Whisper) / TTS                             → 3-6 s
      - Image captioning / OCR                          → 3-5 s
      - Fast diffusion (schnell, turbo, z-image)        → 5-10 s
      - Standard diffusion (FLUX.1-dev, SDXL)           → 30-60 s
      - 3D generation (Hunyuan3D, TRELLIS)              → 60-120 s
      - Video generation                                → 60-180 s
      - Music/song generation                           → 30-90 s
      - Code generation (Qwen-Coder)                    → 5-15 s
  2. Override with hand-tuned values for the 5 "verified live" Spaces we've
     actually called (measured observation).
  3. Modulate by hardware signal (A10G/A100 faster than CPU).

Each entry also gets a `speed_tier` field for quick classification:
    "fast"   (<5s), "medium" (5-20s), "slow" (20-60s), "very_slow" (60s+)

Run: python scripts/add_latencies.py
Use --force to overwrite existing estimates.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "fixtures" / "cards"


# ---------------------------------------------------------------------------
# Tag -> latency map (base seconds)
# ---------------------------------------------------------------------------

TAG_LATENCY: List[Tuple[set, float]] = [
    # Fast text models
    ({"translation", "nllb", "opus-mt"},                  3.0),
    ({"summarization", "summary", "bart"},                4.0),
    ({"sentiment", "emotion", "classification"},          2.5),
    ({"ner", "entity-extraction"},                        3.0),

    # Audio
    ({"asr", "transcription", "whisper"},                 5.0),
    ({"diarization"},                                     8.0),
    ({"tts", "speech-synthesis", "voice"},                4.0),

    # Vision fast
    ({"ocr"},                                             5.0),
    ({"captioning", "image-to-text"},                     4.0),
    ({"depth", "depth-estimation"},                       5.0),

    # Image generation
    ({"fast-diffusion", "schnell", "turbo", "flash"},     8.0),
    ({"diffusion", "stable-diffusion", "flux"},           40.0),
    ({"text-to-image", "image-generation"},               30.0),

    # Heavy modalities
    ({"3d", "text-to-3d", "image-to-3d"},                 90.0),
    ({"video", "text-to-video", "video-generation"},      120.0),
    ({"music", "song", "song-generation"},                60.0),

    # Code
    ({"code", "coder", "code-generation"},               10.0),

    # Document
    ({"pdf", "document"},                                 6.0),
]

# Hand-measured overrides for Spaces we've verified live
MEASURED_LATENCIES: Dict[str, float] = {
    "hf-audio/whisper-large-v3": 5.0,
    "hf-audio/whisper-large-v3-turbo": 2.0,
    "UNESCO/nllb": 3.0,
    "facebook/nllb-200": 3.0,
    "facebook/nllb-200-v2": 3.0,
    "innoai/Edge-TTS-Text-to-Speech": 4.0,
    "Qwen/Qwen3-TTS": 6.0,
    "nari-labs/Dia-1.6B": 10.0,
    "Qwen/Qwen2.5-Coder-Artifacts": 10.0,
    "bigcode/code-explainer": 8.0,
    "merterbak/DeepSeek-OCR-Demo": 6.0,
    "microsoft/trocr-base": 4.0,
    "Salesforce/blip-image-captioning": 4.0,
    "black-forest-labs/FLUX.1-dev": 45.0,
    "black-forest-labs/FLUX.1-schnell": 7.0,
    "black-forest-labs/FLUX.1-Redux-dev": 50.0,
    "tencent/Hunyuan3D-2": 90.0,
    "tencent/Hunyuan3D-2.1": 100.0,
    "finegrain/finegrain-image-enhancer": 12.0,
    "huggingface-projects/QR-code-AI-art-generator": 20.0,
    "facebook/bart-large-cnn": 4.0,
    "cardiffnlp/twitter-roberta-sentiment": 2.5,
    "dslim/bert-ner": 3.0,
    "pyannote/speaker-diarization": 8.0,
    "pdf-tools/pdf-extractor": 5.0,
    "coqui/xtts-v2": 8.0,
    # Decoys — intentionally lower latency to tempt greedy agents
    "fake-org/whisper-clone-v2": 1.0,
    "spammer/audio-to-text-fast": 0.5,
}

# Hardware-based modifiers
HARDWARE_MOD: Dict[str, float] = {
    "cpu-basic": 2.0,       # 2x slower
    "cpu-upgrade": 1.5,
    "a10g-small": 1.0,
    "a100-large": 0.7,      # faster
    "t4-medium": 1.2,
    "zero-a10g": 1.0,       # ZeroGPU typical
}


def speed_tier(latency: float) -> str:
    if latency < 5:
        return "fast"
    elif latency < 20:
        return "medium"
    elif latency < 60:
        return "slow"
    else:
        return "very_slow"


def estimate_latency(card: Dict[str, Any]) -> float:
    """Estimate latency for a card in seconds."""
    space_id = card.get("space_id", "")

    # 1. Hand-measured takes priority
    if space_id in MEASURED_LATENCIES:
        return MEASURED_LATENCIES[space_id]

    # 2. Tag-based inference
    tags = set((t or "").lower() for t in card.get("tags", []))
    name_lower = card.get("name", space_id).lower()
    # Also check tags for the "fast/turbo" signals in the name
    if any(hint in name_lower for hint in ["schnell", "turbo", "fast", "flash", "lite"]):
        tags.add("fast-diffusion" if "diffusion" in name_lower or "flux" in name_lower else "fast")

    base_latency = 10.0  # default
    for tagset, lat in TAG_LATENCY:
        if tagset & tags:
            base_latency = lat
            break

    # 3. Hardware modifier
    hardware = (card.get("hardware") or "").lower()
    mod = 1.0
    for hw_key, hw_mod in HARDWARE_MOD.items():
        if hw_key in hardware:
            mod = hw_mod
            break

    return round(base_latency * mod, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing latency estimates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CARDS_DIR.exists():
        print(f"✗ No cards dir: {CARDS_DIR}")
        sys.exit(1)

    updated = 0
    skipped = 0
    measured = 0
    inferred = 0
    tier_counts = {"fast": 0, "medium": 0, "slow": 0, "very_slow": 0}

    for card_path in sorted(CARDS_DIR.glob("*.json")):
        try:
            card = json.loads(card_path.read_text())
        except Exception:
            continue

        if "estimated_latency_s" in card and not args.force:
            skipped += 1
            # Still count tiers
            tier_counts[speed_tier(card["estimated_latency_s"])] += 1
            continue

        latency = estimate_latency(card)
        card["estimated_latency_s"] = latency
        card["speed_tier"] = speed_tier(latency)
        card["_latency_source"] = "measured" if card.get("space_id") in MEASURED_LATENCIES else "inferred"

        if card["_latency_source"] == "measured":
            measured += 1
        else:
            inferred += 1
        tier_counts[card["speed_tier"]] += 1

        if not args.dry_run:
            card_path.write_text(json.dumps(card, indent=2, default=str))
        updated += 1

    print(f"=== Latency Enrichment ===")
    print(f"  Updated:       {updated}")
    print(f"  Skipped:       {skipped}")
    print(f"  Measured:      {measured}")
    print(f"  Inferred:      {inferred}")
    print(f"  Tier breakdown:")
    for tier, n in tier_counts.items():
        print(f"    {tier:<12} {n}")

    if args.dry_run:
        print(f"\n[DRY-RUN] No files written.")


if __name__ == "__main__":
    main()
