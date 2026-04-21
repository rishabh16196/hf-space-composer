"""
Multi-step GRPO — the real thing, with the model in the env loop at every step.

Why this exists: TRL's `GRPOTrainer` treats each prompt as a one-shot completion.
Our env is multi-step; rewarding "step 0 of LLM then heuristic for the rest"
teaches the policy to act correctly only on initial observations and breaks
its behavior on step 5, 10, 30 (see grpo_local_full regression).

This script runs a custom trajectory-level GRPO:

  1. For each GRPO step:
     a. Sample B task prompts.
     b. For each task, run G independent rollouts where the model generates
        the action at EVERY env step (no heuristic fallback).
     c. Collect (prompt, action_text, step_reward) triples; final trajectory
        reward = episode grade_score.
     d. Compute group-relative advantage per task:
          A_g = (R_g - mean(R_group)) / (std(R_group) + eps)
        Assign A_g to every (prompt, action) pair in that trajectory.
     e. REINFORCE-style update:
          L = -E[A * log π(action | prompt)]  +  β * KL(π || π_ref)
        where π_ref = the SFT adapter weights snapshot taken at start.

Memory cost: two LoRA adapters live in memory simultaneously (trainable + ref
snapshot). On a 128 GB unified memory M-series Mac with Qwen 1.5B bf16, both
adapters together are <1 GB — trivial.

Usage:
    cd local_training
    .venv/bin/python grpo_multistep.py --dry-run
    .venv/bin/python grpo_multistep.py --max-steps 30 --num-gens 4 --batch-size 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent  # spaces_pipeline_env/
sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = (
    "You are an AI agent that orchestrates HuggingFace Spaces to complete "
    "tasks. Output ONLY the next action as JSON with keys 'action_type' "
    "and 'payload'."
)


# ---------------------------------------------------------------------------
# Prompt + action parsing (must match SFT training format)
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
    if obs.auditor_flags:
        parts.append("## Recent Auditor flags:")
        for f in obs.auditor_flags[-3:]:
            parts.append(f"  [{f.get('severity')}] {f.get('message')}")
    if obs.detected_drift:
        parts.append(f"## Detected drift: {obs.detected_drift[-1].get('hint', '')}")
    if obs.recent_outputs:
        parts.append("## Recent outputs:")
        for h in obs.recent_outputs[-3:]:
            parts.append(f"  step {h.get('step')}: success={h.get('success')} | {h.get('output_snippet', '')[:80]}")
    if obs.last_search_results:
        parts.append("## Last search results:")
        for r in obs.last_search_results[:5]:
            parts.append(f"  - {r.get('space_id')} (likes={r.get('likes', 0)}): {r.get('summary', '')[:80]}")
    if obs.last_card_read:
        card = obs.last_card_read
        parts.append(f"## Last card read: {card.get('space_id')}")
        parts.append(f"Description: {(card.get('description') or '')[:200]}")
        parts.append(f"Input schema: {json.dumps(card.get('input_schema', {}), default=str)[:300]}")
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
# Rollout: model in the loop at every env step
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_ids: torch.Tensor      # [seq_len] — chat-templated prompt tokens
    action_ids: torch.Tensor      # [act_len] — generated action tokens
    action_text: str


@dataclass
class Trajectory:
    task_id: str
    steps: List[StepRecord]
    grade: float
    final_steps_taken: int
    n_invalid: int


def _render_prompt(tokenizer, obs: Any) -> torch.Tensor:
    """Render chat-templated prompt → token ids (1D tensor)."""
    user_prompt = build_user_prompt(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids


def rollout_episode(
    env, model, tokenizer, task_id: str,
    max_new_tokens: int = 128, temperature: float = 0.7,
    device: str = "mps", seed: int = 42,
    max_prompt_tokens: int = 2048,
) -> Trajectory:
    """Run one episode, letting `model` generate at every env step.

    Returns a Trajectory with per-step (prompt_ids, action_ids) pairs for
    the policy-gradient update.
    """
    from models import SpacesPipelineAction

    obs = env.reset(seed=seed, task=task_id)
    steps: List[StepRecord] = []
    n_invalid = 0

    model.eval()
    while not obs.done:
        prompt_ids = _render_prompt(tokenizer, obs).to(device)
        if prompt_ids.shape[0] > max_prompt_tokens:
            # Truncate from the left (keep tail, which has "next action" prompt)
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        with torch.no_grad():
            out = model.generate(
                prompt_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0][prompt_ids.shape[0]:]
        action_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        steps.append(StepRecord(
            prompt_ids=prompt_ids.detach().cpu(),
            action_ids=gen_ids.detach().cpu(),
            action_text=action_text,
        ))

        action_dict = parse_action_json(action_text)
        if action_dict is None:
            n_invalid += 1
            # Submit empty → episode ends
            try:
                obs = env.step(SpacesPipelineAction(action_type="submit", payload={"answer": {}}))
            except Exception:
                break
            break

        try:
            action = SpacesPipelineAction(
                action_type=action_dict.get("action_type", "submit"),
                payload=action_dict.get("payload", {}),
            )
        except Exception:
            n_invalid += 1
            break
        obs = env.step(action)

    grade = float(obs.grade_score or 0.0)
    return Trajectory(
        task_id=task_id,
        steps=steps,
        grade=grade,
        final_steps_taken=obs.step_number,
        n_invalid=n_invalid,
    )


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_logprobs(
    model, prompt_ids: torch.Tensor, action_ids: torch.Tensor, device: str,
) -> torch.Tensor:
    """Log-prob of every action token given prompt + prior action tokens.

    Returns a 1D tensor of shape [len(action_ids)].
    """
    prompt_ids = prompt_ids.to(device)
    action_ids = action_ids.to(device)
    full_ids = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0)  # [1, T]
    out = model(full_ids)
    logits = out.logits[0]  # [T, V]
    p_len = prompt_ids.shape[0]
    a_len = action_ids.shape[0]
    action_logits = logits[p_len - 1 : p_len - 1 + a_len]  # [a_len, V]
    logprobs = F.log_softmax(action_logits.float(), dim=-1)
    token_logp = logprobs.gather(1, action_ids.unsqueeze(-1)).squeeze(-1)  # [a_len]
    return token_logp


def compute_logprobs_batched(
    model, records: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str, pad_id: int,
) -> List[torch.Tensor]:
    """Batched version: processes K records in one forward pass via left-padding.

    Returns list of per-record action-token logprobs (shape [a_len_k] each).
    Big speed + memory-utilization win on a GPU with headroom.
    """
    if not records:
        return []

    # Build padded input: each row = [pad...pad, prompt_ids, action_ids]
    # (left-pad so the "last-token-position" for prompt ends at the same column)
    prompt_lens = [p.shape[0] for p, _ in records]
    action_lens = [a.shape[0] for _, a in records]
    seq_lens = [pl + al for pl, al in zip(prompt_lens, action_lens)]
    max_len = max(seq_lens)
    K = len(records)

    input_ids = torch.full((K, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((K, max_len), dtype=torch.long, device=device)
    for i, (p, a) in enumerate(records):
        full = torch.cat([p.to(device), a.to(device)], dim=0)
        input_ids[i, max_len - full.shape[0]:] = full
        attn_mask[i, max_len - full.shape[0]:] = 1

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits  # [K, T, V]

    results: List[torch.Tensor] = []
    for i, (p, a) in enumerate(records):
        p_len, a_len = prompt_lens[i], action_lens[i]
        # Action tokens occupy the last a_len columns; predicted by logits at the
        # a_len positions just before that (logits[t] predicts token t+1).
        pad_prefix = max_len - (p_len + a_len)
        start = pad_prefix + p_len - 1
        action_logits = logits[i, start : start + a_len]  # [a_len, V]
        logprobs = F.log_softmax(action_logits.float(), dim=-1)
        action_ids_dev = a.to(device)
        token_logp = logprobs.gather(1, action_ids_dev.unsqueeze(-1)).squeeze(-1)
        results.append(token_logp)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

HELDOUT_TASKS = {
    "multimodal_caption_speak_024", "multimodal_full_pipeline_025",
    "code_to_speech_020", "doc_quick_summary_015", "audio_sentiment_005",
    "long_doc_localize_032", "long_image_story_033", "long_meeting_analysis_034",
    "marathon_news_evolving_036", "marathon_investigation_037",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/sft_local_1.5b_clean",
                        help="SFT LoRA adapter to start from")
    parser.add_argument("--output-dir", default="outputs/grpo_multistep")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-gens", type=int, default=4,
                        help="Trajectories per task per GRPO step")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Task prompts per GRPO step")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.05,
                        help="KL penalty vs frozen SFT reference")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="Per-action generation budget")
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-micro-batch", type=int, default=8,
                        help="Records per batched forward pass (bigger = more VRAM, faster)")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Log avg rollout reward every N GRPO steps")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N GRPO steps (0 = only at end)")
    parser.add_argument("--n-tasks", type=int, default=20,
                        help="Training task pool (short, non-drift tasks)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(__file__).resolve().parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Multi-step GRPO (trajectory-level) ===")
    print(f"  Device:            {device}")
    print(f"  Model:             {args.model}")
    print(f"  SFT warmstart:     {args.adapter}")
    print(f"  GRPO steps:        {args.max_steps}")
    print(f"  Tasks/step (B):    {args.batch_size}")
    print(f"  Gens/task (G):     {args.num_gens}")
    print(f"  Trajectories/step: {args.batch_size * args.num_gens}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  KL beta:           {args.beta}")
    print(f"  Temperature:       {args.temperature}")

    # --- Task pool ---
    all_tasks = json.loads((ROOT / "fixtures" / "tasks.json").read_text())
    train_ids = [t["task_id"] for t in all_tasks if t["task_id"] not in HELDOUT_TASKS]
    short_ids = [
        t["task_id"] for t in all_tasks
        if t["task_id"] in train_ids
        and t.get("max_actions", 10) <= 15
        and not t.get("drift_events")
    ][: args.n_tasks]
    if not short_ids:
        short_ids = train_ids[: args.n_tasks]
    print(f"  Training task pool ({len(short_ids)}): {short_ids}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model + SFT LoRA (trainable) ---
    print("Loading base model (bf16)...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print(f"  Base loaded in {time.time()-t0:.1f}s")

    adapter_path = args.adapter
    if not os.path.isabs(adapter_path):
        adapter_path = str(Path(__file__).resolve().parent / adapter_path)
    print(f"Loading SFT adapter from {adapter_path} (trainable)...")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_tr:,}")

    # --- Frozen reference: snapshot of SFT adapter weights ---
    # Instead of loading a second model, we snapshot the LoRA adapter weights
    # at start. Then `with disable_adapter_and_use_snapshot` computes ref logp
    # by temporarily swapping weights.
    print("Snapshotting SFT weights as KL reference...")
    ref_state = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if v.requires_grad
    }

    # --- Env + optimizer ---
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment
    env = SpacesPipelineEnvironment()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Testing one rollout...")
        task = short_ids[0]
        traj = rollout_episode(env, model, tokenizer, task,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               device=device, seed=args.seed)
        print(f"  Task: {traj.task_id}")
        print(f"  Steps: {len(traj.steps)}  Grade: {traj.grade:.3f}  Invalid: {traj.n_invalid}")
        if traj.steps:
            print(f"  First action: {traj.steps[0].action_text[:120]}")
        # Test logprob
        s0 = traj.steps[0]
        lp = compute_logprobs(model, s0.prompt_ids, s0.action_ids, device)
        print(f"  Action token logp: shape={tuple(lp.shape)} mean={lp.float().mean().item():.3f}")
        print("\n[DRY-RUN] stopping before training.")
        return

    # --- Training loop ---
    print("\n=== Starting multi-step GRPO ===\n")
    t_train = time.time()
    metrics: List[Dict[str, Any]] = []

    def swap_weights(target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Copy target_state into the trainable params; return old snapshot."""
        old = {}
        for k, v in model.named_parameters():
            if v.requires_grad and k in target_state:
                old[k] = v.detach().clone()
                with torch.no_grad():
                    v.copy_(target_state[k])
        return old

    for step in range(args.max_steps):
        step_t0 = time.time()
        tasks_this_step = random.sample(short_ids, min(args.batch_size, len(short_ids)))
        seed_base = args.seed + step * 1000

        # ---- Phase 1: rollouts (no grad) ----
        all_trajectories: List[List[Trajectory]] = []
        for ti, task in enumerate(tasks_this_step):
            group: List[Trajectory] = []
            for g in range(args.num_gens):
                traj = rollout_episode(
                    env, model, tokenizer, task,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device,
                    seed=seed_base + ti * 100 + g,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                group.append(traj)
            all_trajectories.append(group)

        # ---- Compute advantages (per-task group-relative, with safety clamps) ----
        STD_FLOOR = 0.1     # prevent tiny-variance amplification
        ADV_CLAMP = 3.0     # clamp advantage magnitude to protect against outliers
        flat_records: List[Tuple[StepRecord, float]] = []
        group_stats = []
        n_groups_skipped = 0
        for group in all_trajectories:
            rewards = torch.tensor([t.grade for t in group], dtype=torch.float32)
            mean_r = rewards.mean().item()
            raw_std = rewards.std(unbiased=False).item()
            # Skip groups with no meaningful variance (e.g. all trajectories at 0.15)
            if raw_std < 0.02:
                n_groups_skipped += 1
                group_stats.append((rewards.tolist(), mean_r, 0.0))
                continue
            std_r = max(raw_std, STD_FLOOR)
            group_stats.append((rewards.tolist(), mean_r, std_r))
            for traj in group:
                adv = (traj.grade - mean_r) / std_r
                adv = max(-ADV_CLAMP, min(ADV_CLAMP, adv))
                for sr in traj.steps:
                    flat_records.append((sr, adv))

        if not flat_records:
            print(f"step {step+1:3d}/{args.max_steps}  all groups collapsed to equal rewards — skipping update")
            continue

        # ---- Phase 2: batched policy-gradient update ----
        model.train()
        optim.zero_grad()

        total_loss_val = 0.0
        total_kl_val = 0.0
        n_tokens_total = 0
        MICRO_BATCH = args.update_micro_batch  # K records per forward pass

        # Process records in mini-batches for memory-friendly batched forwards
        n_records = len(flat_records)
        for start in range(0, n_records, MICRO_BATCH):
            batch = flat_records[start : start + MICRO_BATCH]
            batch_records = [(sr.prompt_ids, sr.action_ids) for sr, _ in batch]
            batch_advs = [adv for _, adv in batch]

            # ---- Reference log-probs (no grad, batched) ----
            if args.beta > 0:
                saved = swap_weights(ref_state)
                with torch.no_grad():
                    ref_logps = compute_logprobs_batched(
                        model, batch_records, device, tokenizer.pad_token_id,
                    )
                swap_weights(saved)
                ref_logps = [r.detach() for r in ref_logps]
            else:
                ref_logps = [None] * len(batch)

            # ---- New log-probs (with grad, batched) ----
            new_logps = compute_logprobs_batched(
                model, batch_records, device, tokenizer.pad_token_id,
            )

            # ---- Loss: REINFORCE + KL, summed over records in this micro-batch ----
            batch_loss_terms = []
            for new_lp, ref_lp, adv in zip(new_logps, ref_logps, batch_advs):
                if ref_lp is None:
                    ref_lp = new_lp.detach()
                kl_per_tok = new_lp - ref_lp
                pg_per_tok = -adv * new_lp
                loss_per_tok = pg_per_tok + args.beta * kl_per_tok
                batch_loss_terms.append(loss_per_tok.mean())
                total_kl_val += kl_per_tok.mean().item()
                n_tokens_total += new_lp.shape[0]

            micro_loss = torch.stack(batch_loss_terms).mean()
            # Scale so total step loss = mean across all records (not just this micro-batch)
            scale = len(batch) / n_records
            (micro_loss * scale).backward()
            total_loss_val += micro_loss.item() * len(batch)

        total_loss_val = total_loss_val / n_records

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optim.step()

        # ---- Log ----
        avg_reward = sum(sum(g[0]) / len(g[0]) for g in group_stats) / len(group_stats)
        max_reward = max(max(g[0]) for g in group_stats)
        min_reward = min(min(g[0]) for g in group_stats)
        avg_loss = total_loss_val / max(1, len(flat_records))
        avg_kl = total_kl_val / max(1, len(flat_records))
        elapsed = time.time() - step_t0
        n_steps_in_traj = sum(len(t.steps) for g in all_trajectories for t in g)
        n_invalid = sum(t.n_invalid for g in all_trajectories for t in g)
        print(
            f"step {step+1:3d}/{args.max_steps}  "
            f"avg_r={avg_reward:.3f} [min={min_reward:.2f} max={max_reward:.2f}]  "
            f"n_traj={sum(len(g) for g in all_trajectories)} "
            f"n_steps={n_steps_in_traj} invalid={n_invalid} "
            f"grp_skip={n_groups_skipped}  "
            f"loss={avg_loss:.4f} kl={avg_kl:+.4f}  "
            f"{elapsed:.1f}s"
        )
        metrics.append({
            "step": step + 1,
            "avg_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "loss": avg_loss,
            "kl": avg_kl,
            "n_steps_in_traj": n_steps_in_traj,
            "n_invalid": n_invalid,
            "elapsed_s": elapsed,
        })

        # Periodic checkpoint
        if args.save_every > 0 and (step + 1) % args.save_every == 0 and (step + 1) < args.max_steps:
            ckpt_dir = out_dir / f"checkpoint-{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  ✓ checkpoint saved to {ckpt_dir.name}")

    train_elapsed = time.time() - t_train

    # --- Save adapter ---
    print(f"\n✓ Saving adapter to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Total training time: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")
    print(f"  Metrics: {out_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
