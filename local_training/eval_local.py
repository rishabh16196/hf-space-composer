"""
Local 3-agent comparison on held-out tasks.

Runs the same tasks through:
  1. HeuristicAgent  — gold_pipeline follower (upper bound ceiling)
  2. BaseQwenAgent   — base Qwen 2.5 0.5B Instruct, no training (lower bound)
  3. SftQwenAgent    — base + our SFT LoRA adapter (post-training)

Uses in-process env (no HTTP server needed). Deterministic seed=42.
Greedy decoding for reproducibility.

Usage:
    cd local_training
    .venv/bin/python eval_local.py
    .venv/bin/python eval_local.py --adapter outputs/sft_local --tasks heldout
    .venv/bin/python eval_local.py --tasks real_demo_audio_to_speech
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent  # spaces_pipeline_env/
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Held-out task set (5 tasks, all domains, includes one drift task)
# ---------------------------------------------------------------------------

HELDOUT_TASKS = [
    "audio_sentiment_005",
    "doc_quick_summary_015",
    "code_to_speech_020",
    "multimodal_caption_speak_024",
    "multimodal_full_pipeline_025",
]

SYSTEM_PROMPT = (
    "You are an AI agent that orchestrates HuggingFace Spaces to complete "
    "tasks. Output ONLY the next action as JSON with keys 'action_type' "
    "and 'payload'."
)


# ---------------------------------------------------------------------------
# Prompt builder (must match SFT training format)
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Any) -> str:
    """Copy of scripts/generate_gold_trajectories.py::format_prompt, hint-omitted."""
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
# LLM Agent wrapper (base or SFT)
# ---------------------------------------------------------------------------

class LocalQwenAgent:
    """Loads Qwen 0.5B (optionally with LoRA adapter) and generates actions."""

    def __init__(self, model_id: str, adapter_path: Optional[str] = None,
                 device: Optional[str] = None, max_new_tokens: int = 256):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens

        label = "BaseQwenAgent" if not adapter_path else "SftQwenAgent"
        print(f"[{label}] Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{label}] Loading model (bf16)...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        print(f"[{label}] Model loaded in {time.time()-t0:.1f}s")

        if adapter_path:
            print(f"[{label}] Attaching LoRA adapter from {adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).to(device)

        self.model.eval()

    def reset(self, task_id: str) -> None:
        # Stateless; kept for interface parity with HeuristicAgent
        pass

    def act(self, obs: Any):
        from models import SpacesPipelineAction
        user_prompt = build_user_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        action_dict = parse_action_json(completion)
        if action_dict is None:
            return SpacesPipelineAction(action_type="submit", payload={"answer": {}})
        try:
            return SpacesPipelineAction(
                action_type=action_dict.get("action_type", "submit"),
                payload=action_dict.get("payload", {}),
            )
        except Exception:
            return SpacesPipelineAction(action_type="submit", payload={"answer": {}})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, task_id: str, seed: int = 42) -> Dict[str, Any]:
    obs = env.reset(seed=seed, task=task_id)
    agent.reset(task_id)
    n_invalid = 0
    while not obs.done:
        action = agent.act(obs)
        if action is None:
            break
        obs = env.step(action)
        # Track malformed attempts (NOOP fallback from bad JSON)
        if obs.recent_actions and obs.recent_actions[-1].get("action_type") == "noop":
            n_invalid += 1
    return {
        "task_id": task_id,
        "grade_score": float(obs.grade_score or 0.0),
        "steps": obs.step_number,
        "spaces_called": obs.spaces_called,
        "time_used_s": (obs.grade_details or {}).get("time_used_s", 0),
        "flags": (obs.grade_details or {}).get("flags_count", 0),
        "invalid_json_count": n_invalid,
        "passed": (obs.grade_score or 0) >= 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/sft_local",
                        help="Path to SFT LoRA adapter (relative to local_training/)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs (default: 5 held-out)")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model eval")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT model eval")
    parser.add_argument("--skip-heuristic", action="store_true", help="Skip heuristic eval")
    args = parser.parse_args()

    task_ids = args.tasks or HELDOUT_TASKS

    from server.spaces_pipeline_environment import SpacesPipelineEnvironment
    from inference import HeuristicAgent

    env = SpacesPipelineEnvironment()

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # ---------- Heuristic ----------
    if not args.skip_heuristic:
        print("\n=== HeuristicAgent (follows gold_pipeline) ===")
        agent = HeuristicAgent()
        results = []
        for tid in task_ids:
            r = run_episode(env, agent, tid)
            results.append(r)
            print(f"  {tid:<40} grade={r['grade_score']:.3f}  steps={r['steps']:>3}  flags={r['flags']:>2}")
        all_results["heuristic"] = results

    # ---------- Base Qwen ----------
    if not args.skip_base:
        print(f"\n=== Base Qwen 2.5 0.5B Instruct (no training) ===")
        agent = LocalQwenAgent(args.model, adapter_path=None)
        results = []
        for tid in task_ids:
            t0 = time.time()
            r = run_episode(env, agent, tid)
            r["elapsed_s"] = time.time() - t0
            results.append(r)
            print(f"  {tid:<40} grade={r['grade_score']:.3f}  steps={r['steps']:>3}  invalid_json={r['invalid_json_count']:>2}  {r['elapsed_s']:.0f}s")
        all_results["base_qwen"] = results
        del agent

    # ---------- SFT Qwen ----------
    if not args.skip_sft:
        adapter = args.adapter
        if not os.path.isabs(adapter):
            adapter = str(Path(__file__).resolve().parent / adapter)
        if not Path(adapter).exists():
            print(f"\n✗ SFT adapter not found at {adapter} — skipping")
        else:
            print(f"\n=== SFT Qwen 2.5 0.5B + LoRA adapter ===")
            agent = LocalQwenAgent(args.model, adapter_path=adapter)
            results = []
            for tid in task_ids:
                t0 = time.time()
                r = run_episode(env, agent, tid)
                r["elapsed_s"] = time.time() - t0
                results.append(r)
                print(f"  {tid:<40} grade={r['grade_score']:.3f}  steps={r['steps']:>3}  invalid_json={r['invalid_json_count']:>2}  {r['elapsed_s']:.0f}s")
            all_results["sft_qwen"] = results

    # ---------- Comparison table ----------
    print("\n\n=== COMPARISON ===")
    agents = [a for a in ["base_qwen", "sft_qwen", "heuristic"] if a in all_results]
    header = f"{'Task':<40} " + " ".join(f"{a:<12}" for a in agents)
    print(header)
    print("-" * len(header))
    for i, tid in enumerate(task_ids):
        row = f"{tid:<40} "
        for a in agents:
            if i < len(all_results[a]):
                g = all_results[a][i]["grade_score"]
                row += f"{g:<12.3f}"
        print(row)
    print("-" * len(header))
    row = f"{'AVERAGE':<40} "
    for a in agents:
        avg = sum(r["grade_score"] for r in all_results[a]) / len(all_results[a])
        row += f"{avg:<12.3f}"
    print(row)
    row = f"{'PASS RATE (>=0.5)':<40} "
    for a in agents:
        pr = sum(r["passed"] for r in all_results[a])
        row += f"{pr}/{len(all_results[a]):<11}"
    print(row)

    # Save
    out = Path(__file__).resolve().parent / "outputs" / "eval_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n✓ Saved to {out}")


if __name__ == "__main__":
    main()
