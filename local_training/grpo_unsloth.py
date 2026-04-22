"""
Multi-step GRPO with Unsloth (CUDA-only) — venue-consistent stack.

Design
------
- **One model, two modes**: a single Unsloth `FastLanguageModel` is used for
  both rollouts (switched to inference mode via `for_inference`) and gradient
  updates (switched back via `for_training`). No vLLM, no weight syncing.
- **Batched rollouts**: all active trajectories' prompts are tokenized with
  left-padding and generated in ONE `model.generate()` call per env step.
- **Unsloth kernels**: fused LoRA + triton attention → ~2× faster training and
  significantly faster generation than plain transformers.

This matches the venue target (`scripts/train_grpo.py` is also Unsloth-based),
so results here transfer directly.
"""

from __future__ import annotations

# Unsloth MUST be imported before transformers — patches happen at import time
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SYSTEM_PROMPT = (
    "You are an AI agent that orchestrates HuggingFace Spaces to complete "
    "tasks. Output ONLY the next action as JSON with keys 'action_type' "
    "and 'payload'."
)

HELDOUT_TASKS = {
    "multimodal_caption_speak_024", "multimodal_full_pipeline_025",
    "code_to_speech_020", "doc_quick_summary_015", "audio_sentiment_005",
    "long_doc_localize_032", "long_image_story_033", "long_meeting_analysis_034",
    "marathon_news_evolving_036", "marathon_investigation_037",
}


# ---------------------------------------------------------------------------
# Prompt rendering + action parsing
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


def render_prompt_text(obs: Any, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(obs)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


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
# Rollout — Unsloth batched multi-step
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_ids: torch.Tensor
    action_ids: torch.Tensor
    action_text: str


@dataclass
class Trajectory:
    task_id: str
    steps: List[StepRecord]
    grade: float
    n_env_steps: int
    n_invalid: int


def rollout_batch_unsloth(
    model,
    tokenizer,
    env_class,
    tasks: List[str],
    num_gens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    device: str,
    max_prompt_tokens: int = 2048,
    max_env_iters: int = 40,
) -> List[List[Trajectory]]:
    """Run N=B*G trajectories in parallel using batched Unsloth generate.

    At each iteration, active trajectories are tokenized (left-padded) and
    generated in ONE model.generate() call.
    """
    from models import SpacesPipelineAction

    # Init one runner per (task, gen)
    runners = []
    for ti, tid in enumerate(tasks):
        for g in range(num_gens):
            env = env_class()
            obs = env.reset(seed=base_seed + ti * 1000 + g, task=tid)
            runners.append({
                "task_id": tid, "gen_idx": g,
                "env": env, "obs": obs,
                "steps": [], "n_invalid": 0, "done": False,
            })

    # Ensure tokenizer is left-padded for batched generation
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for _ in range(max_env_iters):
        active = [r for r in runners if not r["done"]]
        if not active:
            break

        prompts = [render_prompt_text(r["obs"], tokenizer) for r in active]
        enc = tokenizer(
            prompts, padding=True, truncation=True,
            max_length=max_prompt_tokens, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = enc.input_ids.shape[1]
        gen_ids_batch = gen[:, prompt_len:]  # [B, L]

        for r, row_input_ids, row_attn, row_gen in zip(
            active, enc.input_ids, enc.attention_mask, gen_ids_batch
        ):
            # Strip left-padding from prompt tokens
            real_prompt_ids = row_input_ids[row_attn.bool()].detach().cpu()
            # Strip trailing pad tokens from generation
            gen_ids = row_gen.detach().cpu()
            # Truncate at first pad
            nonpad_mask = gen_ids != tokenizer.pad_token_id
            if nonpad_mask.any():
                last_nonpad = int(nonpad_mask.nonzero()[-1].item())
                gen_ids = gen_ids[: last_nonpad + 1]
            else:
                gen_ids = gen_ids[:0]

            action_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            r["steps"].append(StepRecord(
                prompt_ids=real_prompt_ids,
                action_ids=gen_ids,
                action_text=action_text,
            ))

            action_dict = parse_action_json(action_text)
            if action_dict is None:
                r["n_invalid"] += 1
                try:
                    r["obs"] = r["env"].step(
                        SpacesPipelineAction(action_type="submit", payload={"answer": {}})
                    )
                except Exception:
                    pass
                r["done"] = True
                continue

            try:
                action = SpacesPipelineAction(
                    action_type=action_dict.get("action_type", "submit"),
                    payload=action_dict.get("payload", {}),
                )
            except Exception:
                r["n_invalid"] += 1
                r["done"] = True
                continue

            r["obs"] = r["env"].step(action)
            if r["obs"].done:
                r["done"] = True

    tokenizer.padding_side = old_pad_side

    # Group by task
    by_task: Dict[str, List[Trajectory]] = {tid: [] for tid in tasks}
    for r in runners:
        grade = float(r["obs"].grade_score or 0.0)
        by_task[r["task_id"]].append(Trajectory(
            task_id=r["task_id"],
            steps=r["steps"],
            grade=grade,
            n_env_steps=r["obs"].step_number,
            n_invalid=r["n_invalid"],
        ))
    return [by_task[tid] for tid in tasks]


# ---------------------------------------------------------------------------
# Log-prob computation (batched, for training forwards)
# ---------------------------------------------------------------------------

def compute_logprobs_batched(
    model, records: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str, pad_id: int,
) -> List[torch.Tensor]:
    if not records:
        return []
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
    logits = out.logits

    results: List[torch.Tensor] = []
    for i, (p, a) in enumerate(records):
        p_len, a_len = prompt_lens[i], action_lens[i]
        pad_prefix = max_len - (p_len + a_len)
        start = pad_prefix + p_len - 1
        action_logits = logits[i, start : start + a_len]
        logprobs = F.log_softmax(action_logits.float(), dim=-1)
        action_ids_dev = a.to(device)
        token_logp = logprobs.gather(1, action_ids_dev.unsqueeze(-1)).squeeze(-1)
        results.append(token_logp)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", required=True, help="SFT LoRA adapter path")
    parser.add_argument("--output-dir", default="outputs/grpo_unsloth")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-gens", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--update-micro-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use Unsloth 4-bit quant for more memory headroom")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "grpo_unsloth.py requires CUDA"
    device = "cuda"

    out_dir = Path(__file__).resolve().parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== GRPO with Unsloth (single-model, batched generate + train) ===")
    print(f"  Model:             {args.model}")
    print(f"  SFT warmstart:     {args.adapter}")
    print(f"  GRPO steps:        {args.max_steps}")
    print(f"  Tasks/step (B):    {args.batch_size}")
    print(f"  Gens/task (G):     {args.num_gens}")
    print(f"  Trajectories/step: {args.batch_size * args.num_gens}")
    print(f"  4-bit:             {args.load_in_4bit}")

    # Task pool
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
    print(f"  Training pool ({len(short_ids)}): {short_ids[:6]}...")

    # --- Load model + SFT adapter via Unsloth ---
    print("\nLoading Unsloth model + SFT adapter...")
    t0 = time.time()
    adapter_path = args.adapter
    if not os.path.isabs(adapter_path):
        adapter_path = str(Path(__file__).resolve().parent / adapter_path)

    # Unsloth natively loads a base + PEFT adapter in ONE call when you pass
    # the adapter path as model_name. It reads adapter_config.json, pulls the
    # base model (matching `base_model_name_or_path`), and attaches the adapter
    # with correct key names — no manual copy needed.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=args.max_seq_length,
        dtype=None,                # auto bf16 on A100/L40S
        load_in_4bit=args.load_in_4bit,
    )
    # Make LoRA trainable (Unsloth sets modules_to_save etc)
    FastLanguageModel.for_training(model)
    print(f"  loaded base + SFT adapter from {adapter_path}")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_tr:,}")
    print(f"  model ready in {time.time()-t0:.1f}s")

    # --- KL reference snapshot ---
    print("Snapshotting current (post-SFT) weights as KL reference...")
    ref_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}

    # --- Env class for rollouts ---
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment

    # --- Dry run ---
    if args.dry_run:
        print("\n[DRY-RUN] single-task rollout with Unsloth for_inference...")
        FastLanguageModel.for_inference(model)
        t0 = time.time()
        groups = rollout_batch_unsloth(
            model, tokenizer, SpacesPipelineEnvironment,
            tasks=[short_ids[0]], num_gens=args.num_gens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            base_seed=args.seed, device=device,
        )
        print(f"  rollout took {time.time()-t0:.1f}s for {args.num_gens} trajectories")
        for t in groups[0]:
            print(f"    grade={t.grade:.3f}  steps={len(t.steps)}  invalid={t.n_invalid}")
        return

    # --- Optimizer ---
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    def swap_weights(target_state):
        old = {}
        for k, p in model.named_parameters():
            if p.requires_grad and k in target_state:
                old[k] = p.detach().clone()
                with torch.no_grad():
                    p.copy_(target_state[k])
        return old

    # --- Training loop ---
    print("\n=== Starting GRPO ===\n")
    t_train = time.time()
    metrics = []

    STD_FLOOR = 0.1
    ADV_CLAMP = 3.0

    for step in range(args.max_steps):
        step_t0 = time.time()
        tasks_this_step = random.sample(short_ids, min(args.batch_size, len(short_ids)))
        seed_base = args.seed + step * 100_000

        # ---- Phase 1: rollouts (Unsloth inference mode) ----
        FastLanguageModel.for_inference(model)
        roll_t0 = time.time()
        all_groups = rollout_batch_unsloth(
            model, tokenizer, SpacesPipelineEnvironment,
            tasks=tasks_this_step, num_gens=args.num_gens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            base_seed=seed_base, device=device,
        )
        roll_elapsed = time.time() - roll_t0

        # ---- Compute advantages ----
        flat_records: List[Tuple[StepRecord, float]] = []
        group_stats = []
        n_groups_skipped = 0
        for group in all_groups:
            rewards = torch.tensor([t.grade for t in group], dtype=torch.float32)
            mean_r = rewards.mean().item()
            raw_std = rewards.std(unbiased=False).item()
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

        n_traj_total = sum(len(g) for g in all_groups)
        n_env_steps_total = sum(t.n_env_steps for g in all_groups for t in g)
        n_invalid = sum(t.n_invalid for g in all_groups for t in g)
        avg_r = sum(sum(g[0]) / len(g[0]) for g in group_stats) / len(group_stats)
        max_r = max(max(g[0]) for g in group_stats)
        min_r = min(min(g[0]) for g in group_stats)

        if not flat_records:
            print(f"step {step+1:3d}/{args.max_steps}  "
                  f"avg_r={avg_r:.3f} [min={min_r:.2f} max={max_r:.2f}]  "
                  f"n_traj={n_traj_total} n_env={n_env_steps_total} invalid={n_invalid} "
                  f"grp_skip={n_groups_skipped}  ALL FLAT — skip update  "
                  f"roll={roll_elapsed:.1f}s")
            metrics.append({"step": step+1, "avg_reward": avg_r, "rollout_s": roll_elapsed,
                            "skipped": True})
            continue

        # ---- Phase 2: gradient update (Unsloth training mode) ----
        FastLanguageModel.for_training(model)
        model.train()
        upd_t0 = time.time()
        optim.zero_grad()
        total_loss_val = 0.0
        total_kl_val = 0.0
        n_records = len(flat_records)

        for start in range(0, n_records, args.update_micro_batch):
            batch = flat_records[start : start + args.update_micro_batch]
            batch_recs = [(sr.prompt_ids, sr.action_ids) for sr, _ in batch]
            batch_advs = [adv for _, adv in batch]

            if args.beta > 0:
                saved = swap_weights(ref_state)
                with torch.no_grad():
                    ref_logps = compute_logprobs_batched(
                        model, batch_recs, device, tokenizer.pad_token_id,
                    )
                swap_weights(saved)
                ref_logps = [r.detach() for r in ref_logps]
            else:
                ref_logps = [None] * len(batch)

            new_logps = compute_logprobs_batched(
                model, batch_recs, device, tokenizer.pad_token_id,
            )

            loss_terms = []
            for new_lp, ref_lp, adv in zip(new_logps, ref_logps, batch_advs):
                if ref_lp is None:
                    ref_lp = new_lp.detach()
                kl_per_tok = new_lp - ref_lp
                pg_per_tok = -adv * new_lp
                loss_per_tok = pg_per_tok + args.beta * kl_per_tok
                loss_terms.append(loss_per_tok.mean())
                total_kl_val += kl_per_tok.mean().item()

            micro_loss = torch.stack(loss_terms).mean()
            scale = len(batch) / n_records
            (micro_loss * scale).backward()
            total_loss_val += micro_loss.item() * len(batch)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optim.step()

        avg_loss = total_loss_val / max(1, n_records)
        avg_kl = total_kl_val / max(1, n_records)
        upd_elapsed = time.time() - upd_t0
        step_elapsed = time.time() - step_t0

        print(
            f"step {step+1:3d}/{args.max_steps}  "
            f"avg_r={avg_r:.3f} [min={min_r:.2f} max={max_r:.2f}]  "
            f"n_traj={n_traj_total} n_env={n_env_steps_total} invalid={n_invalid} "
            f"grp_skip={n_groups_skipped}  "
            f"loss={avg_loss:.4f} kl={avg_kl:+.4f}  "
            f"roll={roll_elapsed:.1f}s upd={upd_elapsed:.1f}s tot={step_elapsed:.1f}s"
        )
        metrics.append({
            "step": step+1, "avg_reward": avg_r,
            "min_reward": min_r, "max_reward": max_r,
            "loss": avg_loss, "kl": avg_kl,
            "n_traj": n_traj_total, "n_env_steps": n_env_steps_total,
            "n_invalid": n_invalid, "grp_skip": n_groups_skipped,
            "rollout_s": roll_elapsed, "update_s": upd_elapsed,
            "step_s": step_elapsed,
        })

        if args.save_every > 0 and (step+1) % args.save_every == 0 and (step+1) < args.max_steps:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  ✓ checkpoint saved → {ckpt.name}")

    # Final save
    print(f"\n✓ Saving final adapter to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    total_train = time.time() - t_train
    print(f"  Total training time: {total_train:.0f}s ({total_train/60:.1f} min)")


if __name__ == "__main__":
    main()
