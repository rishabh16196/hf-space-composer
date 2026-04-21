"""
Multi-step GRPO with vLLM rollouts (CUDA-only).

Design
------
- **Rollouts**: vLLM engine handles ALL generation. All G trajectories per task
  run in parallel (one prompt per active trajectory per env step), batched
  within a single `llm.generate()` call. This is the big win vs sequential
  `model.generate()` — on A100 we see ~10-30× faster rollout phases.
- **Training**: plain transformers + PEFT (bf16). After each GRPO optimizer
  step, save the trainable LoRA adapter to a versioned directory; vLLM picks
  up the new weights on next rollout via `LoRARequest(..., new_path)`.

Memory on a100-large (80 GB):
  - vLLM base weights bf16 (1.5B)  ≈ 3 GB
  - vLLM LoRA adapters cache        ≈ 0.1 GB
  - vLLM KV cache (default budget)  ≈ 25 GB (tune with `gpu_memory_utilization`)
  - training base + PEFT + optim    ≈ 4 GB
  - activations                     ≈ 5 GB at micro-batch=16
  Total ~ 40 GB (comfortable).

Usage
-----
  python grpo_vllm.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter /path/to/sft_adapter \
    --output-dir outputs/grpo_vllm \
    --max-steps 100 --num-gens 6 --batch-size 4 --update-micro-batch 16
"""

from __future__ import annotations

import argparse
import copy
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# vLLM import — must happen after TOKENIZERS_PARALLELISM is set
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
    user_prompt = build_user_prompt(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
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
# Rollout — vLLM batched multi-step
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_ids: torch.Tensor    # CPU long tensor
    action_ids: torch.Tensor
    action_text: str


@dataclass
class Trajectory:
    task_id: str
    steps: List[StepRecord]
    grade: float
    n_env_steps: int
    n_invalid: int


def rollout_batch_vllm(
    llm: LLM,
    env_class,
    tokenizer,
    tasks: List[str],
    num_gens: int,
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest],
    base_seed: int,
    max_env_steps_hard_cap: int = 40,
) -> List[List[Trajectory]]:
    """Run `num_gens` parallel trajectories per task using vLLM.

    Returns a list (per task) of lists (per gen) of Trajectories.
    """
    from models import SpacesPipelineAction

    # Initialize runner state: one per (task, gen) pair
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

    iter_count = 0
    while True:
        active = [r for r in runners if not r["done"]]
        if not active:
            break
        if iter_count >= max_env_steps_hard_cap:
            # Safety: if something loops, force-end remaining runners
            for r in active:
                r["done"] = True
            break
        iter_count += 1

        # Render prompts for all active trajectories in ONE batched vLLM call
        prompt_texts = [render_prompt_text(r["obs"], tokenizer) for r in active]
        outputs = llm.generate(
            prompt_texts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        for r, out in zip(active, outputs):
            out0 = out.outputs[0]
            action_text = out0.text
            prompt_ids = torch.tensor(out.prompt_token_ids, dtype=torch.long)
            action_ids = torch.tensor(out0.token_ids, dtype=torch.long)

            r["steps"].append(StepRecord(
                prompt_ids=prompt_ids,
                action_ids=action_ids,
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

    # Collect trajectories, grouped per task
    by_task: Dict[str, List[Trajectory]] = {tid: [] for tid in tasks}
    for r in runners:
        grade = float(r["obs"].grade_score or 0.0)
        traj = Trajectory(
            task_id=r["task_id"],
            steps=r["steps"],
            grade=grade,
            n_env_steps=r["obs"].step_number,
            n_invalid=r["n_invalid"],
        )
        by_task[r["task_id"]].append(traj)
    return [by_task[tid] for tid in tasks]


# ---------------------------------------------------------------------------
# Log-prob computation (batched, for training forwards)
# ---------------------------------------------------------------------------

def compute_logprobs_batched(
    model, records: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str, pad_id: int,
) -> List[torch.Tensor]:
    """Batched forward with left-padding. Returns list of per-record action logprobs."""
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
    parser.add_argument("--adapter", required=True, help="SFT LoRA adapter to start from")
    parser.add_argument("--output-dir", default="outputs/grpo_vllm")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-gens", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Task prompts per GRPO step")
    parser.add_argument("--update-micro-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55,
                        help="vLLM KV cache fraction (rest reserved for training)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "grpo_vllm.py requires CUDA"
    device = "cuda"

    out_dir = Path(__file__).resolve().parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Directory where we save versioned adapter snapshots for vLLM to load
    adapter_stage_root = out_dir / "adapters"
    adapter_stage_root.mkdir(parents=True, exist_ok=True)

    print("=== GRPO with vLLM rollouts + PEFT training ===")
    print(f"  Model:             {args.model}")
    print(f"  SFT warmstart:     {args.adapter}")
    print(f"  GRPO steps:        {args.max_steps}")
    print(f"  Tasks/step (B):    {args.batch_size}")
    print(f"  Gens/task (G):     {args.num_gens}")
    print(f"  Trajectories/step: {args.batch_size * args.num_gens}")
    print(f"  LR:                {args.lr}")
    print(f"  KL beta:           {args.beta}")
    print(f"  vLLM KV mem frac:  {args.gpu_memory_utilization}")

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

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Training model (transformers + PEFT, bf16) ---
    print("Loading base model (bf16) for training...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print(f"  base loaded in {time.time()-t0:.1f}s")

    adapter_path = args.adapter
    if not os.path.isabs(adapter_path):
        adapter_path = str(Path(__file__).resolve().parent / adapter_path)
    print(f"Loading SFT adapter from {adapter_path} (trainable)...")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_tr:,}")

    # --- Reference snapshot for KL ---
    print("Snapshotting SFT weights as KL reference...")
    ref_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}

    # --- vLLM engine (separate) ---
    print("Loading vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2,
        max_cpu_loras=4,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=3072,
    )
    print(f"  vLLM loaded in {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
    )

    # Stage initial adapter (copy SFT as version 1) for vLLM
    def stage_adapter(version: int) -> str:
        dst = adapter_stage_root / f"v{version:04d}"
        if dst.exists():
            shutil.rmtree(dst)
        # Save current trainable LoRA
        model.save_pretrained(str(dst))
        return str(dst)

    v = 1
    cur_adapter_path = stage_adapter(v)
    def lora_req_for(version: int, path: str) -> LoRARequest:
        return LoRARequest(f"spacesgrpo-v{version}", version, path)

    # --- Env class for rollouts ---
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment

    # --- Dry-run path ---
    if args.dry_run:
        print("\n[DRY-RUN] single-task rollout test...")
        t0 = time.time()
        groups = rollout_batch_vllm(
            llm=llm, env_class=SpacesPipelineEnvironment, tokenizer=tokenizer,
            tasks=[short_ids[0]], num_gens=args.num_gens,
            sampling_params=sampling_params, lora_request=lora_req_for(v, cur_adapter_path),
            base_seed=args.seed,
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
    print("\n=== Starting GRPO (vLLM rollouts) ===\n")
    t_train = time.time()
    metrics = []

    STD_FLOOR = 0.1
    ADV_CLAMP = 3.0

    for step in range(args.max_steps):
        step_t0 = time.time()
        tasks_this_step = random.sample(short_ids, min(args.batch_size, len(short_ids)))
        seed_base = args.seed + step * 100_000

        # ---- Phase 1: vLLM rollouts ----
        roll_t0 = time.time()
        all_groups = rollout_batch_vllm(
            llm=llm, env_class=SpacesPipelineEnvironment, tokenizer=tokenizer,
            tasks=tasks_this_step, num_gens=args.num_gens,
            sampling_params=sampling_params,
            lora_request=lora_req_for(v, cur_adapter_path),
            base_seed=seed_base,
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

        # Stats for logging
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
                  f"grp_skip={n_groups_skipped}  ALL GROUPS FLAT — skipping update  "
                  f"roll={roll_elapsed:.1f}s")
            metrics.append({"step": step+1, "avg_reward": avg_r, "rollout_s": roll_elapsed,
                            "skipped": True})
            continue

        # ---- Phase 2: batched PG update ----
        upd_t0 = time.time()
        model.train()
        optim.zero_grad()
        total_loss_val = 0.0
        total_kl_val = 0.0
        n_records = len(flat_records)

        for start in range(0, n_records, args.update_micro_batch):
            batch = flat_records[start : start + args.update_micro_batch]
            batch_recs = [(sr.prompt_ids, sr.action_ids) for sr, _ in batch]
            batch_advs = [adv for _, adv in batch]

            # ref logp (no grad, batched, with weight swap)
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

            # new logp (with grad)
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

        # ---- Stage new adapter version for next rollout ----
        stage_t0 = time.time()
        v += 1
        cur_adapter_path = stage_adapter(v)
        # Clean up old adapter dirs to save disk (keep last 2)
        old_dirs = sorted(adapter_stage_root.glob("v*"))
        for d in old_dirs[:-2]:
            shutil.rmtree(d, ignore_errors=True)
        stage_elapsed = time.time() - stage_t0

        step_elapsed = time.time() - step_t0
        print(
            f"step {step+1:3d}/{args.max_steps}  "
            f"avg_r={avg_r:.3f} [min={min_r:.2f} max={max_r:.2f}]  "
            f"n_traj={n_traj_total} n_env={n_env_steps_total} invalid={n_invalid} "
            f"grp_skip={n_groups_skipped}  "
            f"loss={avg_loss:.4f} kl={avg_kl:+.4f}  "
            f"roll={roll_elapsed:.1f}s upd={upd_elapsed:.1f}s stage={stage_elapsed:.1f}s "
            f"tot={step_elapsed:.1f}s"
        )
        metrics.append({
            "step": step+1, "avg_reward": avg_r,
            "min_reward": min_r, "max_reward": max_r,
            "loss": avg_loss, "kl": avg_kl,
            "n_traj": n_traj_total, "n_env_steps": n_env_steps_total,
            "n_invalid": n_invalid, "grp_skip": n_groups_skipped,
            "rollout_s": roll_elapsed, "update_s": upd_elapsed,
            "stage_s": stage_elapsed, "step_s": step_elapsed,
        })

        if args.save_every > 0 and (step+1) % args.save_every == 0 and (step+1) < args.max_steps:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  ✓ checkpoint saved → {ckpt.name}")

    # --- Save final adapter ---
    print(f"\n✓ Saving final adapter to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    total_train = time.time() - t_train
    print(f"  Total training time: {total_train:.0f}s ({total_train/60:.1f} min)")


if __name__ == "__main__":
    main()
