"""
Two-tier held-out evaluation: easy vs hard.

  EASY tier — 5 mid-difficulty tasks (single-domain or short multi-domain).
              Measures "did SFT produce a valid agent following format + gold pipeline?"
  HARD tier — 5 long-horizon / marathon tasks (7+ steps, schema drift).
              Measures "does training actually solve the hard OpenEnv problem?"

Reports separate averages + pass rates per tier for HeuristicAgent, Base Qwen,
and SFT Qwen, so the pitch narrative can distinguish "valid agent" from
"actually solves the hard problem."

Usage:
    cd local_training
    .venv/bin/python eval_two_tier.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter outputs/sft_local_1.5b_clean
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse LocalQwenAgent + runner from eval_local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_local import LocalQwenAgent, run_episode  # noqa: E402


EASY_HELDOUT = [
    "audio_sentiment_005",
    "doc_quick_summary_015",
    "code_to_speech_020",
    "multimodal_caption_speak_024",
    "multimodal_full_pipeline_025",
]

HARD_HELDOUT = [
    "long_doc_localize_032",
    "long_image_story_033",
    "long_meeting_analysis_034",
    "marathon_news_evolving_036",
    "marathon_investigation_037",
]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {"avg": 0.0, "pass": 0, "n": 0}
    avg = sum(r["grade_score"] for r in results) / len(results)
    passed = sum(1 for r in results if r["passed"])
    return {"avg": avg, "pass": passed, "n": len(results)}


def run_tier(env, agent, task_ids: List[str], label: str) -> List[Dict[str, Any]]:
    results = []
    for tid in task_ids:
        t0 = time.time()
        r = run_episode(env, agent, tid)
        r["elapsed_s"] = time.time() - t0
        results.append(r)
        print(f"  [{label}] {tid:<42} grade={r['grade_score']:.3f}  "
              f"steps={r['steps']:>3}  invalid={r.get('invalid_json_count', 0):>2}  "
              f"{r['elapsed_s']:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/sft_local_1.5b_clean")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-heuristic", action="store_true")
    args = parser.parse_args()

    from server.spaces_pipeline_environment import SpacesPipelineEnvironment
    from inference import HeuristicAgent

    env = SpacesPipelineEnvironment()
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # ---------- Heuristic ----------
    if not args.skip_heuristic:
        print("\n=== HeuristicAgent ===")
        agent = HeuristicAgent()
        all_results["heuristic"] = {
            "easy": run_tier(env, agent, EASY_HELDOUT, "easy"),
            "hard": run_tier(env, agent, HARD_HELDOUT, "hard"),
        }

    # ---------- Base Qwen ----------
    if not args.skip_base:
        print(f"\n=== Base {args.model} (no training) ===")
        agent = LocalQwenAgent(args.model, adapter_path=None)
        all_results["base_qwen"] = {
            "easy": run_tier(env, agent, EASY_HELDOUT, "easy"),
            "hard": run_tier(env, agent, HARD_HELDOUT, "hard"),
        }
        del agent

    # ---------- SFT Qwen ----------
    if not args.skip_sft:
        adapter = args.adapter
        if not os.path.isabs(adapter):
            adapter = str(Path(__file__).resolve().parent / adapter)
        if not Path(adapter).exists():
            print(f"\n✗ SFT adapter not found at {adapter} — skipping")
        else:
            print(f"\n=== SFT {args.model} + LoRA adapter ===")
            agent = LocalQwenAgent(args.model, adapter_path=adapter)
            all_results["sft_qwen"] = {
                "easy": run_tier(env, agent, EASY_HELDOUT, "easy"),
                "hard": run_tier(env, agent, HARD_HELDOUT, "hard"),
            }

    # ---------- Comparison table ----------
    print("\n\n=== TWO-TIER SUMMARY ===")
    agents = [a for a in ["base_qwen", "sft_qwen", "heuristic"] if a in all_results]
    header = f"{'Tier':<10} " + " ".join(f"{a:<22}" for a in agents)
    print(header)
    print("-" * len(header))
    for tier in ["easy", "hard"]:
        row = f"{tier.upper():<10} "
        for a in agents:
            s = summarize(all_results[a][tier])
            row += f"avg={s['avg']:.3f} pass={s['pass']}/{s['n']:<6} "
        print(row)

    # Combined
    print("-" * len(header))
    row = f"{'ALL(10)':<10} "
    for a in agents:
        combined = all_results[a]["easy"] + all_results[a]["hard"]
        s = summarize(combined)
        row += f"avg={s['avg']:.3f} pass={s['pass']}/{s['n']:<6} "
    print(row)

    # Save
    out = Path(__file__).resolve().parent / "outputs" / "eval_two_tier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n✓ Saved to {out}")


if __name__ == "__main__":
    main()
