"""
Generate gold training trajectories by running the HeuristicAgent on tasks.

The HeuristicAgent follows each task's `gold_pipeline`, which produces ~0.94
grade on average. Each step yields (observation_prompt, action_json, reward).

Output formats produced:
  - fixtures/gold_trajectories.jsonl
      Full trajectories: one JSON object per EPISODE with all steps + final grade
  - fixtures/sft_pairs.jsonl
      Flattened (prompt, completion) pairs — drop-in for HF TRL SFTTrainer
  - fixtures/grpo_prompts.jsonl
      Task prompts only (for GRPO which generates its own rollouts)

Usage:
    # Default: all training tasks, 3 seeds each
    python scripts/generate_gold_trajectories.py

    # Specific tasks
    python scripts/generate_gold_trajectories.py --tasks real_demo_audio_to_speech

    # More seeds for data augmentation
    python scripts/generate_gold_trajectories.py --seeds 10

    # Dry-run (print a sample, don't write)
    python scripts/generate_gold_trajectories.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


FIXTURES_DIR = ROOT / "fixtures"
TRAJ_FILE = FIXTURES_DIR / "gold_trajectories.jsonl"
SFT_FILE = FIXTURES_DIR / "sft_pairs.jsonl"
GRPO_PROMPTS_FILE = FIXTURES_DIR / "grpo_prompts.jsonl"


# ---------------------------------------------------------------------------
# Split: which tasks are for training vs held-out evaluation
# ---------------------------------------------------------------------------

HELDOUT_TASKS = {
    "multimodal_caption_speak_024",
    "multimodal_full_pipeline_025",
    "code_to_speech_020",
    "doc_quick_summary_015",
    "audio_sentiment_005",
}


def format_prompt(obs: Any) -> str:
    """Format an observation as a prompt the LLM will see at training time.

    This has to match EXACTLY how the LLM is prompted at inference time.
    See inference.py::LLMAgent._build_prompt for the reference format.
    """
    parts = [
        f"## Task: {obs.task_description}",
        f"Input: {json.dumps(obs.task_input, default=str)[:400]}",
        f"Expected output schema: {json.dumps(obs.expected_output_schema, default=str)}",
        f"Step {obs.step_number}/{obs.max_steps}, actions remaining: {obs.actions_remaining}, space budget: {obs.spaces_budget_remaining}",
    ]
    if obs.expert_persona_hint:
        parts.append(f"Expert hint: {obs.expert_persona_hint}")
    if obs.auditor_flags:
        recent = obs.auditor_flags[-3:]
        parts.append("## Recent Auditor flags:")
        for f in recent:
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


def format_action(action) -> str:
    """Serialize an action back to JSON (the LLM's expected output format)."""
    return json.dumps(
        {"action_type": action.action_type, "payload": action.payload},
        default=str,
    )


def run_one_episode(env, agent, task_id: str, seed: int) -> Dict[str, Any]:
    """Run a single episode, collect (prompt, action, reward) per step."""
    obs = env.reset(seed=seed, task=task_id)
    agent.reset(task_id)

    steps: List[Dict[str, Any]] = []
    total_reward = 0.0

    while not obs.done:
        prompt = format_prompt(obs)
        action = agent.act(obs)
        if action is None:
            break
        action_json = format_action(action)

        obs = env.step(action)
        reward = float(obs.reward or 0.0)
        total_reward += reward

        steps.append({
            "step": obs.step_number,
            "prompt": prompt,
            "action": action_json,
            "reward": round(reward, 4),
            "done": obs.done,
        })

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "n_steps": len(steps),
        "total_reward": round(total_reward, 4),
        "grade_score": float(obs.grade_score or 0.0),
        "grade_details": obs.grade_details,
        "passed": bool((obs.grade_score or 0.0) >= 0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs (default: all training tasks)")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per task (default: 3)")
    parser.add_argument("--include-heldout", action="store_true",
                        help="Include held-out tasks (default: skip them)")
    parser.add_argument("--min-grade", type=float, default=0.5,
                        help="Skip trajectories with grade below this (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Import env + agent (must happen after sys.path setup above)
    from inference import HeuristicAgent
    from server.spaces_pipeline_environment import SpacesPipelineEnvironment

    # Load task list
    tasks_path = FIXTURES_DIR / "tasks.json"
    all_tasks = json.loads(tasks_path.read_text())
    all_task_ids = [t["task_id"] for t in all_tasks]

    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = [t for t in all_task_ids
                    if args.include_heldout or t not in HELDOUT_TASKS]

    print(f"Generating gold trajectories")
    print(f"  Tasks:   {len(task_ids)} {'(inc held-out)' if args.include_heldout else '(train split only)'}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Total:   {len(task_ids) * args.seeds} episodes")
    print(f"  Min grade filter: {args.min_grade}")
    print()

    # Run episodes
    env = SpacesPipelineEnvironment()
    agent = HeuristicAgent()

    trajectories: List[Dict[str, Any]] = []
    sft_pairs: List[Dict[str, Any]] = []
    grpo_prompts: List[Dict[str, Any]] = []

    n_run = 0
    n_passed = 0
    n_filtered = 0
    t0 = time.time()

    for task_id in task_ids:
        for seed in range(42, 42 + args.seeds):
            traj = run_one_episode(env, agent, task_id, seed)
            n_run += 1
            if traj["passed"]:
                n_passed += 1
            if traj["grade_score"] < args.min_grade:
                n_filtered += 1
                continue

            trajectories.append(traj)

            # Flatten into SFT pairs
            for step in traj["steps"]:
                sft_pairs.append({
                    "task_id": task_id,
                    "seed": seed,
                    "step": step["step"],
                    "prompt": step["prompt"],
                    "completion": step["action"],
                    "reward": step["reward"],
                    "grade_score": traj["grade_score"],
                })

            # GRPO prompt (just task-level, no step prompts)
            grpo_prompts.append({
                "task_id": task_id,
                "prompt": (
                    f"## Task: {traj['steps'][0]['prompt'].split('Task:', 1)[-1].split('##')[0].strip() if traj['steps'] else task_id}"
                ),
            })

    elapsed = time.time() - t0

    print(f"=== Results ===")
    print(f"  Episodes run:         {n_run}")
    print(f"  Passed (grade>=0.5):  {n_passed}")
    print(f"  Filtered (low grade): {n_filtered}")
    print(f"  Kept trajectories:    {len(trajectories)}")
    print(f"  SFT pairs:            {len(sft_pairs)}")
    print(f"  GRPO prompts:         {len(set(p['task_id'] for p in grpo_prompts))} unique tasks")
    print(f"  Time:                 {elapsed:.1f}s")

    if args.dry_run:
        print("\n[DRY-RUN] Sample trajectory:")
        if trajectories:
            t = trajectories[0]
            print(f"  task_id: {t['task_id']}, seed: {t['seed']}, grade: {t['grade_score']}")
            print(f"  first step prompt (truncated):\n    {t['steps'][0]['prompt'][:300]}...")
            print(f"  first step action: {t['steps'][0]['action']}")
        print("\n[DRY-RUN] No files written.")
        return

    # Write trajectories (episode-level)
    TRAJ_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAJ_FILE, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"\n✓ Wrote {len(trajectories)} episodes to {TRAJ_FILE.name}")

    # Write SFT pairs (step-level)
    with open(SFT_FILE, "w") as f:
        for p in sft_pairs:
            f.write(json.dumps(p, default=str) + "\n")
    print(f"✓ Wrote {len(sft_pairs)} SFT pairs to {SFT_FILE.name}")

    # Write GRPO prompts (task-level)
    unique_prompts = {p["task_id"]: p for p in grpo_prompts}
    with open(GRPO_PROMPTS_FILE, "w") as f:
        for tid, p in unique_prompts.items():
            f.write(json.dumps(p, default=str) + "\n")
    print(f"✓ Wrote {len(unique_prompts)} GRPO prompts to {GRPO_PROMPTS_FILE.name}")


if __name__ == "__main__":
    main()
