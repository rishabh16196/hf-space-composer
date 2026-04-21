"""
Evaluate an agent (trained or baseline) against held-out tasks.

Held-out evaluation tasks (last 5 of 25):
    multimodal_caption_speak_024
    multimodal_full_pipeline_025
    code_to_speech_020
    doc_quick_summary_015
    audio_sentiment_005

Usage:
    # Baseline heuristic
    python scripts/evaluate.py --agent heuristic

    # Trained checkpoint
    python scripts/evaluate.py --agent trained --model-path ./outputs/phase4

    # LLM via OpenAI API (zero-shot)
    OPENAI_API_KEY=... python scripts/evaluate.py --agent llm --model gpt-4o-mini
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


HOLDOUT_TASKS = [
    "multimodal_caption_speak_024",
    "multimodal_full_pipeline_025",
    "code_to_speech_020",
    "doc_quick_summary_015",
    "audio_sentiment_005",
]


async def evaluate_agent(agent_name: str, env_url: str, n_episodes: int = 1) -> Dict[str, Any]:
    """Run agent on each held-out task, collect grades."""
    from spaces_pipeline_env import SpacesPipelineEnv

    if agent_name == "heuristic":
        from inference import HeuristicAgent
        agent = HeuristicAgent()
    elif agent_name == "llm":
        from inference import LLMAgent
        agent = LLMAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        )
    elif agent_name == "trained":
        # Stub: would load PEFT adapter
        print("Trained agent loader not implemented in this build")
        sys.exit(1)
    else:
        print(f"Unknown agent: {agent_name}")
        sys.exit(1)

    env = SpacesPipelineEnv(base_url=env_url)
    await env.connect()

    results: List[Dict[str, Any]] = []

    try:
        for task_id in HOLDOUT_TASKS:
            for ep in range(n_episodes):
                print(f"\n=== {task_id} (ep {ep+1}/{n_episodes}) ===")
                result = await env.reset(task=task_id, seed=42 + ep)
                obs = result.observation
                agent.reset(task_id)

                while not result.done:
                    action = agent.act(obs)
                    if action is None:
                        break
                    result = await env.step(action)
                    obs = result.observation

                grade = obs.grade_score or 0.0
                details = obs.grade_details or {}
                print(f"  Grade: {grade:.3f} | Components: {details.get('components', {})}")
                results.append({
                    "task_id": task_id,
                    "episode": ep,
                    "grade": grade,
                    "details": details,
                })
    finally:
        await env.close()

    avg_grade = sum(r["grade"] for r in results) / len(results) if results else 0.0
    pass_rate = sum(1 for r in results if r["grade"] >= 0.5) / len(results) if results else 0.0

    print(f"\n=== Summary ===")
    print(f"  Average grade: {avg_grade:.3f}")
    print(f"  Pass rate (>=0.5): {pass_rate:.1%}")
    return {"avg_grade": avg_grade, "pass_rate": pass_rate, "results": results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="heuristic", choices=["heuristic", "llm", "trained"])
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", help="Save results to JSON")
    args = parser.parse_args()

    results = await evaluate_agent(args.agent, args.env_url, args.episodes)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
