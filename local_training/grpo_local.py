"""
Local GRPO smoke test — Apple Silicon (MPS) edition.

Purpose: prove the GRPO machinery works end-to-end on a laptop.
NOT expected to produce a meaningfully better agent — MPS is ~50x slower
than a CUDA+vLLM setup, so this is a correctness/plumbing test, not a
real RL training run.

Pipeline:
  1. Load SFT LoRA adapter (or base model) in bf16
  2. For each training-split task, build an initial-observation prompt
  3. GRPOTrainer samples N completions per prompt at temperature>0
  4. Reward function: parse completion as first action, run episode
     (heuristic fills in the rest), return grade_score as reward
  5. GRPO gradient step pushes model toward higher-reward first actions

Real training uses scripts/train_grpo.py on CUDA+Unsloth at the venue.

Usage:
    cd local_training
    .venv/bin/python grpo_local.py --dry-run                    # sanity check
    .venv/bin/python grpo_local.py --max-steps 3 --num-gens 2   # 3-step smoke
    .venv/bin/python grpo_local.py --max-steps 20               # longer run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# GRPO on MPS: no vllm, no flash-attn; force CPU-safe fallbacks
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parent.parent  # spaces_pipeline_env/
sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = (
    "You are an AI agent that orchestrates HuggingFace Spaces to complete "
    "tasks. Output ONLY the next action as JSON with keys 'action_type' "
    "and 'payload'."
)


# ---------------------------------------------------------------------------
# Prompt builder (matches eval_local.build_user_prompt)
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Any) -> str:
    parts = [
        f"## Task: {obs.task_description}",
        f"Input: {json.dumps(obs.task_input, default=str)[:400]}",
        f"Expected output schema: {json.dumps(obs.expected_output_schema, default=str)}",
        f"Step {obs.step_number}/{obs.max_steps}, actions remaining: {obs.actions_remaining}, space budget: {obs.spaces_budget_remaining}",
    ]
    if obs.expert_persona_hint:
        parts.append(f"Expert hint: {obs.expert_persona_hint}")
    parts.append("\n## Your next action (JSON only):")
    return "\n".join(parts)


def parse_action_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Reward function (runs an env episode per completion)
# ---------------------------------------------------------------------------

_env_singleton = None
_heuristic_singleton = None


def _get_env_and_heuristic():
    global _env_singleton, _heuristic_singleton
    if _env_singleton is None:
        from server.spaces_pipeline_environment import SpacesPipelineEnvironment
        from inference import HeuristicAgent
        _env_singleton = SpacesPipelineEnvironment()
        _heuristic_singleton = HeuristicAgent()
    return _env_singleton, _heuristic_singleton


def run_episode_reward(task_id: str, completion: str, seed: int = 42) -> float:
    """Use `completion` as the first action; heuristic finishes the episode.

    Returns grade_score in [0,1]. Invalid JSON → 0.0 (below engagement gate 0.15).
    """
    from models import SpacesPipelineAction
    env, heuristic = _get_env_and_heuristic()

    obs = env.reset(seed=seed, task=task_id)
    action_dict = parse_action_json(completion)
    if action_dict is None:
        # Submit empty to terminate — env will grade → engagement gate caps at 0.15 or lower
        try:
            bad_action = SpacesPipelineAction(action_type="submit", payload={"answer": {}})
            obs = env.step(bad_action)
        except Exception:
            return 0.0
        return float(obs.grade_score or 0.0)

    try:
        action = SpacesPipelineAction(
            action_type=action_dict.get("action_type", "submit"),
            payload=action_dict.get("payload", {}),
        )
    except Exception:
        return 0.0

    obs = env.step(action)

    # Complete with heuristic
    heuristic.reset(task_id)
    while not obs.done:
        follow = heuristic.act(obs)
        if follow is None:
            break
        obs = env.step(follow)

    return float(obs.grade_score or 0.0)


# TRL 1.2.0 reward_fn signature: reward_fn(completions, **kwargs) where kwargs
# carries every column from the dataset (including 'prompt' and our 'task_id').
def build_reward_fn(verbose: bool = True):
    call_count = {"n": 0}

    def reward_fn(completions, **kwargs) -> List[float]:
        # `task_id` is passed through because it's a dataset column
        task_ids = kwargs.get("task_id", [])
        if not isinstance(task_ids, list):
            task_ids = [task_ids] * len(completions)
        # Completions may be a list of strings OR list of chat-turn dicts.
        rewards: List[float] = []
        for i, c in enumerate(completions):
            if isinstance(c, list):
                # chat format: take the assistant's last content
                text = c[-1].get("content", "") if c else ""
            else:
                text = c
            tid = task_ids[i] if i < len(task_ids) else task_ids[0]
            try:
                r = run_episode_reward(tid, text)
            except Exception as e:
                if verbose:
                    print(f"  [reward_fn] task={tid} FAILED: {e}")
                r = 0.0
            rewards.append(r)
        call_count["n"] += 1
        if verbose:
            avg = sum(rewards) / max(1, len(rewards))
            print(f"  [reward_fn call #{call_count['n']}] batch_size={len(rewards)} avg_reward={avg:.3f} min={min(rewards):.3f} max={max(rewards):.3f}")
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset: one row per (task, seed) — the prompt is the initial obs
# ---------------------------------------------------------------------------

def build_grpo_dataset(task_ids: List[str], tokenizer, seed_base: int = 42, seeds_per_task: int = 1) -> Dataset:
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment
    env = SpacesPipelineEnvironment()

    rows = []
    for tid in task_ids:
        for s in range(seeds_per_task):
            obs = env.reset(seed=seed_base + s, task=tid)
            user_prompt = build_user_prompt(obs)
            # Render through chat template so the policy sees its training format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            rows.append({"prompt": prompt_text, "task_id": tid})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HELDOUT_TASKS = {
    # Match generate_gold_trajectories.py exactly
    "multimodal_caption_speak_024", "multimodal_full_pipeline_025",
    "code_to_speech_020", "doc_quick_summary_015", "audio_sentiment_005",
    "long_doc_localize_032", "long_image_story_033", "long_meeting_analysis_034",
    "marathon_news_evolving_036", "marathon_investigation_037",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/sft_local_1.5b_clean",
                        help="SFT LoRA adapter to warm-start from (or 'none' for base)")
    parser.add_argument("--output-dir", default="outputs/grpo_local_smoke")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="GRPO optimization steps (keep TINY on MPS)")
    parser.add_argument("--num-gens", type=int, default=2,
                        help="Completions per prompt (min 2 for GRPO variance)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Prompts per step; must be divisible by num_gens")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Action JSONs are short (<200 tokens)")
    parser.add_argument("--n-tasks", type=int, default=6,
                        help="How many training tasks to sample for the prompt set")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty. Higher = policy stays closer to SFT (safer on MPS)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / args.output_dir

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Local GRPO smoke test ===")
    print(f"  Device:             {device}")
    print(f"  Model:              {args.model}")
    print(f"  SFT warmstart:      {args.adapter}")
    print(f"  Max GRPO steps:     {args.max_steps}")
    print(f"  Num generations:    {args.num_gens}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Completion length:  {args.max_completion_length}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  Output:             {out_dir}")

    # ---- Build prompt dataset from training-split tasks ----
    all_tasks = json.loads((ROOT / "fixtures" / "tasks.json").read_text())
    train_task_ids = [t["task_id"] for t in all_tasks if t["task_id"] not in HELDOUT_TASKS]
    # Prefer short/simple tasks for the smoke test — marathons are too slow for reward rollout
    short_tasks = [
        t["task_id"] for t in all_tasks
        if t["task_id"] in train_task_ids
        and t.get("max_actions", 10) <= 15
        and not t.get("drift_events")
    ][:args.n_tasks]
    if not short_tasks:
        short_tasks = train_task_ids[:args.n_tasks]
    print(f"  Training tasks ({len(short_tasks)}): {short_tasks}")

    # ---- Tokenizer ----
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_grpo_dataset(short_tasks, tokenizer)
    print(f"  Dataset size: {len(dataset)} prompts")
    print(f"  First prompt (truncated):\n    {dataset[0]['prompt'][:200]}...")

    # ---- Model + LoRA adapter ----
    print("\nLoading model (bf16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print(f"  Base loaded in {time.time()-t0:.1f}s")

    adapter_path = args.adapter
    if adapter_path and adapter_path.lower() != "none":
        if not os.path.isabs(adapter_path):
            adapter_path = str(Path(__file__).resolve().parent / adapter_path)
        if Path(adapter_path).exists():
            print(f"Loading SFT LoRA from {adapter_path} (trainable)...")
            model = PeftModel.from_pretrained(
                model, adapter_path, is_trainable=True,
            ).to(device)
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_tr:,}")
        else:
            print(f"  ⚠ adapter not found at {adapter_path} — attaching fresh LoRA")
            from peft import get_peft_model
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

    # ---- GRPO config ----
    # MPS/CPU: keep everything tiny
    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_gens,
        max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=0.95,
        logging_steps=1,
        save_steps=max(5, args.max_steps),
        save_total_limit=1,
        bf16=(device != "cpu"),
        fp16=False,
        beta=args.beta,           # KL penalty vs reference (higher = safer on MPS)
        report_to="none",
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        max_grad_norm=0.5,        # tight grad clip — MPS bf16 sampling hates big updates
        remove_unused_columns=False,  # keep task_id column accessible to reward_fn
    )

    print(f"\n  Steps to run: {args.max_steps}")
    print(f"  Expected calls to reward_fn: {args.max_steps} × {args.num_gens} × {args.batch_size} = {args.max_steps*args.num_gens*args.batch_size} rollouts")

    if args.dry_run:
        print("\n[DRY-RUN] Testing reward_fn on one prompt...")
        reward_fn = build_reward_fn(verbose=True)
        sample_completion = '{"action_type": "search_spaces", "payload": {"query": "whisper speech to text", "top_k": 3}}'
        r = reward_fn([sample_completion], task_id=[short_tasks[0]])
        print(f"  Sample reward: {r}")
        print("\n[DRY-RUN] Stopping before training.")
        return

    reward_fn = build_reward_fn(verbose=True)

    print("\n=== Starting GRPO ===\n")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    t0 = time.time()
    try:
        trainer.train()
        elapsed = time.time() - t0
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print(f"\n✓ Saved GRPO adapter to {out_dir}")
        print(f"  Total GRPO time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted — saving current state...")
        trainer.save_model(str(out_dir))
        print(f"  Saved partial adapter to {out_dir}")


if __name__ == "__main__":
    main()
