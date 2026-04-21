"""
Supervised Fine-Tuning (SFT) warmstart on gold trajectories.

Trains the base model (Qwen 2.5 1.5B) to imitate the HeuristicAgent's behavior
before switching to GRPO. This gives the model a "tool-use shape" — it learns
to output valid JSON actions — so GRPO starts from a much better baseline.

Input: fixtures/sft_pairs.jsonl (and optionally llm_sft_pairs.jsonl)
Output: outputs/sft_warmstart/ — LoRA checkpoint

Stack: Unsloth (fast 4-bit training) + TRL SFTTrainer.

Typical settings:
  - 348 SFT pairs × 3 epochs = ~1000 gradient steps
  - Free Colab T4: 15-30 min
  - A100: 5 min

Usage:
    # Dry-run (validate without installing heavy deps)
    python scripts/sft_warmstart.py --dry-run

    # Full training
    python scripts/sft_warmstart.py --output-dir outputs/sft_warmstart --epochs 3

    # Include LLM-generated trajectories
    python scripts/sft_warmstart.py --include-llm
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = ROOT / "fixtures"


def load_sft_pairs(include_llm: bool = False) -> List[Dict[str, Any]]:
    """Load SFT pairs from heuristic + optionally LLM trajectories."""
    pairs: List[Dict[str, Any]] = []
    heuristic_file = FIXTURES_DIR / "sft_pairs.jsonl"
    if heuristic_file.exists():
        with open(heuristic_file) as f:
            for line in f:
                pairs.append(json.loads(line))

    if include_llm:
        llm_file = FIXTURES_DIR / "llm_sft_pairs.jsonl"
        if llm_file.exists():
            with open(llm_file) as f:
                for line in f:
                    pairs.append(json.loads(line))

    return pairs


def build_chat_messages(prompt: str, completion: str) -> List[Dict[str, str]]:
    """Format as a chat conversation for modern Instruct models."""
    system = (
        "You are an AI agent that orchestrates HuggingFace Spaces to complete tasks. "
        "Output ONLY the next action as JSON with keys 'action_type' and 'payload'."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/sft_warmstart")
    parser.add_argument("--include-llm", action="store_true",
                        help="Also use LLM-generated trajectories")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load data
    pairs = load_sft_pairs(include_llm=args.include_llm)
    print(f"Loaded {len(pairs)} SFT pairs")
    if not pairs:
        print("✗ No SFT pairs found. Run scripts/generate_gold_trajectories.py first.")
        sys.exit(1)

    # Quick stats
    grades = [p["grade_score"] for p in pairs if "grade_score" in p]
    print(f"  Avg grade in pairs: {sum(grades)/len(grades):.3f}" if grades else "")

    if args.dry_run:
        print("\n[DRY-RUN] Setup check:")
        try:
            import trl  # noqa
            print("  ✓ trl installed")
        except ImportError:
            print("  ✗ trl not installed — pip install trl")
        try:
            import unsloth  # noqa
            print("  ✓ unsloth installed")
        except ImportError:
            print("  ⚠ unsloth not installed — pip install unsloth (optional)")

        print("\n[DRY-RUN] Sample formatted chat message:")
        sample = pairs[0]
        msgs = build_chat_messages(sample["prompt"], sample["completion"])
        for m in msgs:
            role = m["role"]
            content = m["content"]
            print(f"  [{role}] {content[:200]}" + ("..." if len(content) > 200 else ""))

        print(f"\n[DRY-RUN] Training config preview:")
        print(f"  Model: {args.model}")
        print(f"  LoRA r: {args.lora_r}")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Effective batch: {args.batch_size * args.grad_accum}")
        print(f"  Total examples: {len(pairs)}")
        print(f"  Estimated steps: {len(pairs) // (args.batch_size * args.grad_accum) * args.epochs}")
        return

    # Real training path — requires heavy deps
    try:
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"✗ Missing deps for training: {e}")
        print("  Install with: pip install -e '.[training]' trl datasets torch")
        sys.exit(1)

    # Try Unsloth for speed, fall back to plain transformers
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer

    # Prepare dataset
    dataset_records = []
    for p in pairs:
        msgs = build_chat_messages(p["prompt"], p["completion"])
        dataset_records.append({"messages": msgs})
    train_dataset = Dataset.from_list(dataset_records)
    print(f"✓ Built dataset: {len(train_dataset)} examples")

    # Load model
    if use_unsloth:
        print("Loading model with Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r * 2,
            bias="none",
        )
    else:
        print("Loading model with transformers (full precision)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )

    # Train
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting SFT...\n")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Saved SFT-warmstart model to {args.output_dir}")


if __name__ == "__main__":
    main()
