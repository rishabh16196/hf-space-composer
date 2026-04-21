"""
Generate gold trajectories by running an LLM solver (GPT-5 / Claude 4.5 / Qwen
72B) via the OpenAI-compatible API.

Use this when HeuristicAgent trajectories aren't diverse/realistic enough — LLM
trajectories include backtracks, retries, verbose reasoning, which all make for
better SFT training data than the heuristic's "always follow gold pipeline"
behavior.

Writes to fixtures/llm_trajectories.jsonl (additive — appends to prior runs).

Usage:
    # Set API key first
    export OPENAI_API_KEY=hf_xxxx    # or sk-...
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

    # Run on specific tasks
    python scripts/generate_llm_trajectories.py --tasks audio_summarize_hindi_001 --episodes 3

    # Keep only passing trajectories
    python scripts/generate_llm_trajectories.py --tasks all --min-grade 0.5

Cost warning: each task ~ 10-20 LLM calls. 25 tasks × 3 seeds = ~1500 calls.
Plan for $1-10 depending on model.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = ROOT / "fixtures"
TRAJ_FILE = FIXTURES_DIR / "llm_trajectories.jsonl"
SFT_FILE = FIXTURES_DIR / "llm_sft_pairs.jsonl"


async def run_llm_episode(env, llm_agent, task_id: str, seed: int, format_prompt_fn):
    """Run an episode with the LLM agent, record per-step data."""
    result = await env.reset(task=task_id, seed=seed)
    obs = result.observation
    llm_agent.reset(task_id)

    steps: List[Dict[str, Any]] = []
    total_reward = 0.0

    while not result.done:
        prompt = format_prompt_fn(obs)
        action = llm_agent.act(obs)
        if action is None:
            break

        result = await env.step(action)
        obs = result.observation
        reward = float(result.reward or 0.0)
        total_reward += reward

        steps.append({
            "step": obs.step_number,
            "prompt": prompt,
            "action": json.dumps({"action_type": action.action_type, "payload": action.payload}, default=str),
            "reward": round(reward, 4),
            "done": result.done,
        })

    return {
        "task_id": task_id,
        "seed": seed,
        "model": os.getenv("MODEL_NAME", "unknown"),
        "steps": steps,
        "n_steps": len(steps),
        "total_reward": round(total_reward, 4),
        "grade_score": float(obs.grade_score or 0.0),
        "passed": bool((obs.grade_score or 0.0) >= 0.5),
    }


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs (or 'all'); default: 5 common ones")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per task (seeds 42, 43, ...)")
    parser.add_argument("--min-grade", type=float, default=0.0,
                        help="Only save trajectories with grade >= this")
    parser.add_argument("--env-url", default="http://localhost:8000",
                        help="Running SpacesPipelineEnv server URL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max trajectories to generate (hard cap)")
    args = parser.parse_args()

    from inference import LLMAgent, _build_user_prompt
    from spaces_pipeline_env import SpacesPipelineEnv

    # Build format_prompt from inference.LLMAgent._build_prompt for parity
    # Actually inference.LLMAgent already builds prompts internally; we'll use
    # a simplified copy to record them.
    # Import directly for trajectory recording:
    from generate_gold_trajectories import format_prompt

    # Task list
    all_tasks = json.loads((FIXTURES_DIR / "tasks.json").read_text())
    all_ids = [t["task_id"] for t in all_tasks]
    if args.tasks and args.tasks != ["all"]:
        task_ids = args.tasks
    elif args.tasks == ["all"]:
        task_ids = all_ids
    else:
        # Default: 5 representative tasks
        task_ids = [
            "real_demo_audio_to_speech",
            "audio_summarize_hindi_001",
            "image_caption_translate_006",
            "doc_extract_summarize_011",
            "code_explain_translate_016",
        ]
    print(f"Tasks: {task_ids}")
    print(f"Episodes per task: {args.episodes}")

    # Connect env
    env = SpacesPipelineEnv(base_url=args.env_url)
    await env.connect()

    agent = LLMAgent()
    if not agent.client:
        print("✗ No LLM credentials (set OPENAI_API_KEY / HF_TOKEN)")
        sys.exit(1)

    print(f"Using model: {agent.model} at {agent.base_url}")
    print()

    # Open output file in append mode
    TRAJ_FILE.parent.mkdir(parents=True, exist_ok=True)
    traj_f = open(TRAJ_FILE, "a")
    sft_f = open(SFT_FILE, "a")

    n_run = 0
    n_kept = 0
    t0 = time.time()

    try:
        for task_id in task_ids:
            for seed in range(42, 42 + args.episodes):
                if args.limit and n_run >= args.limit:
                    break
                print(f"\n[{n_run+1}] Running {task_id} seed={seed}...")
                try:
                    traj = await run_llm_episode(env, agent, task_id, seed, format_prompt)
                except Exception as e:
                    print(f"  ✗ failed: {e}")
                    n_run += 1
                    continue

                n_run += 1
                print(f"  grade={traj['grade_score']:.3f} steps={traj['n_steps']}")

                if traj["grade_score"] < args.min_grade:
                    print(f"  skipped (below min grade {args.min_grade})")
                    continue

                traj_f.write(json.dumps(traj, default=str) + "\n")
                traj_f.flush()

                for step in traj["steps"]:
                    sft_f.write(json.dumps({
                        "task_id": task_id,
                        "seed": seed,
                        "model": traj["model"],
                        "step": step["step"],
                        "prompt": step["prompt"],
                        "completion": step["action"],
                        "reward": step["reward"],
                        "grade_score": traj["grade_score"],
                    }, default=str) + "\n")
                sft_f.flush()

                n_kept += 1
            if args.limit and n_run >= args.limit:
                break

        elapsed = time.time() - t0
        print(f"\n=== Summary ===")
        print(f"  Ran:    {n_run}")
        print(f"  Kept:   {n_kept}")
        print(f"  Time:   {elapsed:.0f}s")
        print(f"  Wrote to: {TRAJ_FILE}")
        print(f"  SFT pairs: {SFT_FILE}")

    finally:
        traj_f.close()
        sft_f.close()
        await env.close()


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
