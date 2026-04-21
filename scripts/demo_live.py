"""
Live demo runner — pretty-prints agent reasoning + Auditor + Expert events.

Use this for the hackathon pitch. Renders a clean three-pane view:
  Top:     Task description + budget
  Middle:  Agent action + Space output snippet
  Bottom:  Auditor flags + Expert feedback

Toggle live mode with SPACES_MODE=live.

Usage:
    python scripts/demo_live.py --task multimodal_caption_speak_024 --agent hybrid
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def render_banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def render_step(step_num: int, action, output_snippet: str, reward: float, flags) -> None:
    print(f"\n--- Step {step_num} ---")
    print(f"  Agent action:  {action.action_type}")
    payload_preview = str(action.payload)[:110]
    print(f"  Payload:       {payload_preview}")
    print(f"  Output:        {(output_snippet or '(none)')[:110]}")
    print(f"  Reward:        {reward:+.3f}")
    if flags:
        print(f"  Auditor:")
        for f in flags[-3:]:
            print(f"    [{f.get('severity'):<8}] {f.get('message', '')[:90]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="audio_summarize_hindi_001")
    parser.add_argument("--agent", default="hybrid", choices=["heuristic", "llm", "hybrid"])
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--mode", default="mock", choices=["mock", "live", "record"],
                        help="Sets SPACES_MODE before launch")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Sleep between steps (for visual demo pacing)")
    args = parser.parse_args()

    os.environ["SPACES_MODE"] = args.mode

    from inference import HeuristicAgent, HybridAgent, LLMAgent
    from spaces_pipeline_env import SpacesPipelineEnv

    if args.agent == "heuristic":
        agent = HeuristicAgent()
    elif args.agent == "llm":
        agent = LLMAgent()
    else:
        agent = HybridAgent()

    env = SpacesPipelineEnv(base_url=args.env_url)
    await env.connect()

    try:
        result = await env.reset(task=args.task)
        obs = result.observation
        agent.reset(args.task)

        render_banner(f"TASK: {obs.task_id}  |  MODE: {args.mode.upper()}  |  AGENT: {args.agent}")
        print(f"  Description: {obs.task_description}")
        print(f"  Budget: {obs.actions_remaining} actions, {obs.spaces_budget_remaining} Space calls")
        print(f"  Persona hint: {obs.expert_persona_hint}")

        step_num = 0
        prev_flag_count = 0
        while not result.done and step_num < obs.max_steps:
            step_num += 1
            action = agent.act(obs)
            if action is None:
                break
            result = await env.step(action)
            obs = result.observation
            new_flags = obs.auditor_flags[prev_flag_count:]
            prev_flag_count = len(obs.auditor_flags)
            output_snippet = (obs.recent_outputs[-1].get("output_snippet")
                              if obs.recent_outputs else "")
            render_step(step_num, action, output_snippet, result.reward or 0.0, new_flags)
            if obs.detected_drift and step_num > 0:
                last_drift = obs.detected_drift[-1]
                if last_drift.get("step") == obs.step_number:
                    print(f"  ⚠  DRIFT: {last_drift.get('hint', '')[:80]}")
            if obs.expert_recent_feedback and result.done:
                print(f"\n  Expert feedback: {obs.expert_recent_feedback}")
            time.sleep(args.delay)

        render_banner("RESULT")
        score = obs.grade_score or 0.0
        details = obs.grade_details or {}
        print(f"  Final grade:   {score:.3f}  {'PASS ✓' if score >= 0.5 else 'FAIL ✗'}")
        print(f"  Components:    {details.get('components', {})}")
        print(f"  Persona used:  {details.get('persona', '')}")
        print(f"  Total flags:   {details.get('flags_count', 0)}")
        print(f"  Actions used:  {details.get('actions_used', 0)} / {obs.max_steps}")
        print(f"  Spaces called: {details.get('spaces_called', 0)}")

    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
