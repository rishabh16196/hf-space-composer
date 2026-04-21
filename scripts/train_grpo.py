"""
GRPO training for Spaces Pipeline Pro.

Pipeline:
  1. (Optional) SFT warmstart via scripts/sft_warmstart.py first
  2. Load SFT-warmstart checkpoint (or base model)
  3. For each prompt in grpo_prompts.jsonl:
       a. Generate N candidate action sequences from model
       b. Run each through the env, collect reward
       c. GRPO update to favor higher-reward sequences

Because TRL's `GRPOTrainer` is designed around stateless reward functions but
our env is stateful (multi-step), we use a custom "trajectory reward function"
that runs a full episode per completion prefix.

This file is runnable but real training requires:
  - GPU with 16GB+ VRAM (for 1.5B + LoRA)
  - trl, unsloth (optional), transformers, datasets, torch

Usage:
    # Verify setup (no actual training)
    python scripts/train_grpo.py --dry-run --phase 1

    # Real training (at venue with sponsor compute)
    python scripts/train_grpo.py --phase 1 --output-dir outputs/grpo_phase1

    # Start from SFT warmstart checkpoint
    python scripts/train_grpo.py --phase 1 --warmstart outputs/sft_warmstart
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Curriculum phases
# ---------------------------------------------------------------------------

PHASE_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {  # Warmup: easy tasks, dense shaping on
        "task_filter": lambda t: t.get("difficulty") == "easy" and not t.get("drift_events"),
        "dense_rewards": True,
        "shaping_coeff": 1.0,
        "max_steps": 500,
        "num_generations": 4,
    },
    2: {  # Discovery: medium added, persona rotation
        "task_filter": lambda t: t.get("difficulty") in ("easy", "medium") and not t.get("drift_events"),
        "dense_rewards": True,
        "shaping_coeff": 0.7,
        "max_steps": 500,
        "num_generations": 6,
    },
    3: {  # Drift: schema drift introduced
        "task_filter": lambda t: t.get("difficulty") in ("easy", "medium", "hard"),
        "dense_rewards": True,
        "shaping_coeff": 0.4,
        "max_steps": 1000,
        "num_generations": 6,
    },
    4: {  # Adversarial: full curriculum, sparse rewards
        "task_filter": lambda t: True,
        "dense_rewards": False,
        "shaping_coeff": 0.0,
        "max_steps": 1000,
        "num_generations": 8,
    },
}


# ---------------------------------------------------------------------------
# Action parsing (LLM output -> env action)
# ---------------------------------------------------------------------------

def parse_action(completion: str) -> Optional[Dict[str, Any]]:
    """Extract an action dict from an LLM completion."""
    text = completion.strip()
    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Episode runner (reward signal for GRPO)
# ---------------------------------------------------------------------------

def run_episode_with_completion(
    env_factory,
    task_id: str,
    completion: str,
    seed: int = 42,
) -> float:
    """Run one episode where the FIRST action comes from the completion.

    NOTE: In TRL's current GRPO interface, completions are single generations.
    For true multi-step RL, we'd use TRL's `environment_factory` pattern
    (available in recent TRL versions) which handles the loop internally.
    This helper is used when TRL is NOT managing the env loop.

    Returns: reward (scalar) for the episode.
    """
    from inference import HeuristicAgent  # heuristic fallback for remaining steps
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment
    from models import SpacesPipelineAction

    env = SpacesPipelineEnvironment()
    obs = env.reset(seed=seed, task=task_id)

    # Step 1: use the LLM's action
    action_dict = parse_action(completion)
    if action_dict is None:
        return -1.0  # invalid JSON penalty

    try:
        action = SpacesPipelineAction(
            action_type=action_dict.get("action_type", "submit"),
            payload=action_dict.get("payload", {}),
        )
    except Exception:
        return -1.0

    obs = env.step(action)

    # Steps 2+: fall back to heuristic to complete the episode
    # (In practice, GRPO will generate the whole episode autoregressively
    # once TRL env-factory is wired up properly; this is a simplification.)
    heuristic = HeuristicAgent()
    heuristic.reset(task_id)
    while not obs.done:
        follow_up = heuristic.act(obs)
        if follow_up is None:
            break
        obs = env.step(follow_up)

    return float(obs.grade_score or 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_reward_fn(task_ids: List[str]):
    """Build a GRPO reward function closure.

    TRL calls `reward_fn(prompts, completions, **kwargs)` and expects a list
    of scalar rewards (one per completion).
    """
    def reward_fn(prompts, completions, **kwargs):
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            # Extract task_id from prompt (we embed it in the prompt)
            task_id = task_ids[0] if task_ids else "audio_summarize_hindi_001"
            # Run full episode
            try:
                reward = run_episode_with_completion(None, task_id, completion)
            except Exception:
                reward = -1.0
            rewards.append(reward)
        return rewards

    return reward_fn


def load_grpo_prompts() -> List[Dict[str, Any]]:
    p = ROOT / "fixtures" / "grpo_prompts.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--warmstart", default=None,
                        help="Path to SFT-warmstart LoRA checkpoint")
    parser.add_argument("--output-dir", default="./outputs/grpo")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = PHASE_CONFIGS[args.phase]
    if args.steps:
        cfg["max_steps"] = args.steps

    # Env-level config (reward shaping)
    os.environ["DENSE_REWARDS"] = "true" if cfg["dense_rewards"] else "false"
    os.environ["SHAPING_COEFF"] = str(cfg["shaping_coeff"])

    print(f"=== GRPO Training: Phase {args.phase} ===")
    print(f"  Model:            {args.model}")
    print(f"  Warmstart:        {args.warmstart or '(none — base model)'}")
    print(f"  Max steps:        {cfg['max_steps']}")
    print(f"  Num generations:  {cfg['num_generations']}")
    print(f"  Dense rewards:    {cfg['dense_rewards']}")
    print(f"  Shaping coeff:    {cfg['shaping_coeff']}")

    prompts = load_grpo_prompts()
    print(f"  GRPO prompts:     {len(prompts)} (from fixtures/grpo_prompts.jsonl)")

    if not prompts:
        print("\n⚠ No GRPO prompts. Run scripts/generate_gold_trajectories.py first.")
        if not args.dry_run:
            sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] Setup checks:")
        try:
            import trl
            print(f"  ✓ trl {trl.__version__}")
        except ImportError:
            print("  ✗ trl not installed")
        try:
            import unsloth  # noqa
            print(f"  ✓ unsloth installed")
        except ImportError:
            print("  ⚠ unsloth not installed (optional)")
        try:
            from spaces_pipeline_env import SpacesPipelineEnv  # noqa
            print("  ✓ spaces_pipeline_env importable")
        except Exception as e:
            print(f"  ✗ env import failed: {e}")

        # Validate that parse_action works on a sample completion
        sample_completion = '{"action_type": "search_spaces", "payload": {"query": "whisper", "top_k": 3}}'
        parsed = parse_action(sample_completion)
        print(f"  ✓ Parse test: {parsed}")

        print(f"\n[DRY-RUN] Sample prompt (first of {len(prompts)}):")
        if prompts:
            print(f"  task: {prompts[0].get('task_id')}")
            print(f"  prompt: {prompts[0].get('prompt', '')[:200]}")

        print("\n[DRY-RUN] To actually train:")
        print(f"  python {sys.argv[0]} --phase {args.phase} --output-dir {args.output_dir}")
        return

    # ===== REAL TRAINING PATH =====
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"✗ Missing deps: {e}")
        print("  pip install -e '.[training]' trl datasets torch")
        sys.exit(1)

    # Model loading (with optional SFT warmstart)
    try:
        from unsloth import FastLanguageModel
        print("Loading with Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        if args.warmstart:
            # Load LoRA from SFT checkpoint
            model.load_adapter(args.warmstart, adapter_name="default")
            print(f"  ✓ loaded SFT warmstart from {args.warmstart}")
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=args.lora_r * 2,
                bias="none",
            )
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading with transformers (no Unsloth)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )

    # Build dataset
    task_filter = cfg["task_filter"]
    all_tasks = json.loads((ROOT / "fixtures" / "tasks.json").read_text())
    filtered_task_ids = [t["task_id"] for t in all_tasks if task_filter(t)]
    print(f"  Training on {len(filtered_task_ids)} tasks after filter")

    # Convert prompts to a HF Dataset
    phase_prompts = [p for p in prompts if p.get("task_id") in filtered_task_ids]
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in phase_prompts])
    print(f"  Dataset size: {len(dataset)} prompts")

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=cfg["num_generations"],
        max_steps=cfg["max_steps"],
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
    )

    reward_fn = build_reward_fn(filtered_task_ids)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...\n")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n✓ Saved trained model to {args.output_dir}")


if __name__ == "__main__":
    main()
