"""
Local SFT warmstart — Apple Silicon (MPS) edition.

Minimum-viable training run for the hackathon prototype:
  - Small model (Qwen 2.5 0.5B Instruct by default, Llama 3.2 1B if you have access)
  - LoRA r=16 (parameter-efficient, tiny disk footprint)
  - bf16 on MPS
  - No Unsloth (not supported on macOS), no bitsandbytes (CUDA-only)
  - Uses SFT pairs from ../fixtures/sft_pairs.jsonl (771 prompt/completion pairs)

This script is intentionally separate from scripts/sft_warmstart.py — the
existing script targets Unsloth + CUDA for the hackathon venue. This one is
for prototyping locally.

Usage:
    # From local_training/ directory
    .venv/bin/python sft_local.py                                   # default: Qwen 0.5B, 1 epoch
    .venv/bin/python sft_local.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
    .venv/bin/python sft_local.py --model meta-llama/Llama-3.2-1B-Instruct  # needs HF license accepted
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Quiet down tokenizer warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent  # spaces_pipeline_env/
SFT_FILE = ROOT / "fixtures" / "sft_pairs.jsonl"


SYSTEM_PROMPT = (
    "You are an AI agent that orchestrates HuggingFace Spaces to complete "
    "tasks. Output ONLY the next action as JSON with keys 'action_type' "
    "and 'payload'."
)


def load_sft_pairs() -> List[Dict[str, Any]]:
    if not SFT_FILE.exists():
        raise FileNotFoundError(f"No SFT pairs at {SFT_FILE}. Run scripts/generate_gold_trajectories.py first.")
    pairs = []
    with open(SFT_FILE) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def build_dataset(pairs: List[Dict[str, Any]], tokenizer) -> Dataset:
    """Render each (prompt, completion) pair via the model's chat template."""
    records = []
    for p in pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        records.append({"text": text})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model ID (ungated default; Llama 3.2 1B needs license accepted)")
    parser.add_argument("--output-dir", default="outputs/sft_local")
    parser.add_argument("--epochs", type=int, default=1, help="Default 1 for MVP")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Keep small on MPS to avoid OOM")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1536,
                        help="Keep <2K on MPS for memory headroom")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N SFT pairs (for quick smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load everything, print plan, exit without training")
    args = parser.parse_args()

    # Resolve output path relative to this file, not cwd
    out_dir = Path(__file__).resolve().parent / args.output_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"=== Local SFT (min-viable) ===")
    print(f"  Device:         {device}")
    print(f"  Model:          {args.model}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  LoRA r:         {args.lora_r}")
    print(f"  Max seq len:    {args.max_seq_length}")
    print(f"  Output:         {out_dir}")

    # --- Load data ---
    pairs = load_sft_pairs()
    if args.subset:
        pairs = pairs[:args.subset]
    print(f"  SFT pairs:      {len(pairs)}")
    avg_grade = sum(p.get("grade_score", 0) for p in pairs) / max(1, len(pairs))
    print(f"  Avg grade:      {avg_grade:.3f}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset ---
    print("Building dataset with chat template...")
    train_dataset = build_dataset(pairs, tokenizer)
    print(f"  Dataset size: {len(train_dataset)} examples")
    print(f"  Sample (first 300 chars):\n    {train_dataset[0]['text'][:300]}...")

    # --- Model ---
    print("\nLoading model (bf16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # --- LoRA ---
    print("\nAttaching LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    if args.dry_run:
        print("\n[DRY-RUN] Stopping before training.")
        return

    # --- SFT config ---
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=(device != "cpu"),
        fp16=False,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        warmup_ratio=0.05,
    )

    # --- Train ---
    print("\n=== Starting SFT ===\n")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # --- Save ---
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n✓ Saved LoRA adapter to {out_dir}")
    print(f"  Total training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
