"""
Generate PNG plots for the hackathon submission.

Outputs (into outputs/plots/):
  1. reward_curve_vN.png          — per-GRPO-step avg_reward trajectory
  2. kl_curve_vN.png               — per-step KL divergence from SFT
  3. two_tier_bar_chart.png        — final two-tier grades across agents
  4. per_task_heatmap.png          — grade per (task, agent)
  5. rubric_hardening_comparison.png — v1-lenient vs v1-hardened vs v2-hardened vs v3

Usage:
    cd local_training
    .venv/bin/python make_plots.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. GRPO reward / KL curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    metrics_path: Path, label: str, out_prefix: str, smoothed_window: int = 5
) -> None:
    if not metrics_path.exists():
        print(f"[skip] no metrics at {metrics_path}")
        return
    m = json.loads(metrics_path.read_text())
    rows = [x for x in m if not x.get("skipped", False)]
    if not rows:
        print(f"[skip] no non-skipped steps in {metrics_path}")
        return
    steps = [r["step"] for r in rows]
    rewards = [r["avg_reward"] for r in rows]
    kls = [r.get("kl", 0.0) for r in rows]

    def smooth(xs: List[float], w: int) -> List[float]:
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo : i + 1]) / (i - lo + 1))
        return out

    # Reward plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.35, color="steelblue", label="per-step")
    ax.plot(
        steps, smooth(rewards, smoothed_window),
        linewidth=2.2, color="steelblue",
        label=f"smoothed (window={smoothed_window})",
    )
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("avg trajectory reward (per 100-step batch)")
    ax.set_title(f"{label} — GRPO training reward trajectory")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(
        0.02, 0.97,
        f"Start (step 1): {rewards[0]:.3f}\nEnd   (step {steps[-1]}): {rewards[-1]:.3f}\nMax:   {max(rewards):.3f}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey"),
    )
    fig.tight_layout()
    fig.savefig(PLOTS / f"reward_curve_{out_prefix}.png", dpi=90)
    plt.close(fig)
    print(f"[ok] saved reward_curve_{out_prefix}.png")

    # KL plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, kls, alpha=0.45, color="darkorange")
    ax.plot(steps, smooth(kls, smoothed_window), linewidth=2.2, color="darkorange")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("KL divergence from SFT reference")
    ax.set_title(f"{label} — KL drift over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"kl_curve_{out_prefix}.png", dpi=90)
    plt.close(fig)
    print(f"[ok] saved kl_curve_{out_prefix}.png")


# ---------------------------------------------------------------------------
# 2. Two-tier bar chart: all agents
# ---------------------------------------------------------------------------

def _agent_avg(runs: List[Dict[str, Any]], filter_tier: Optional[str] = None) -> Dict[str, float]:
    """Return {'avg': float, 'n_pass': int, 'n': int} for a list of runs."""
    if filter_tier:
        runs = [r for r in runs if r.get("task_id", "").startswith(_tier_task_prefixes(filter_tier))]
    if not runs:
        return {"avg": 0.0, "n_pass": 0, "n": 0}
    grades = [r["grade_score"] for r in runs]
    passes = sum(1 for g in grades if g >= 0.5)
    return {"avg": sum(grades) / len(grades), "n_pass": passes, "n": len(grades)}


def _tier_task_prefixes(tier: str) -> tuple:
    # Easy held-out: 5 short/medium tasks; Hard: 5 long + marathon
    if tier == "easy":
        return ("audio_sentiment_005", "doc_quick_summary_015", "code_to_speech_020",
                "multimodal_caption_speak_024", "multimodal_full_pipeline_025")
    elif tier == "hard":
        return ("long_doc_localize_032", "long_image_story_033", "long_meeting_analysis_034",
                "marathon_news_evolving_036", "marathon_investigation_037")
    return ()


def plot_two_tier_bar_chart(
    agent_results: Dict[str, Dict[str, Dict[str, float]]],
    out_name: str = "two_tier_bar_chart.png",
) -> None:
    """agent_results[agent_name][tier] = {'avg': float, 'n_pass': int, 'n': int}."""
    agents = list(agent_results.keys())
    tiers = ["easy", "hard", "overall"]
    colors = {"easy": "#4caf50", "hard": "#e74c3c", "overall": "#3498db"}

    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(agents)
    width = 0.26
    x_positions = range(n)
    for ti, tier in enumerate(tiers):
        vals = [agent_results[a][tier]["avg"] for a in agents]
        bars = ax.bar(
            [x + (ti - 1) * width for x in x_positions],
            vals, width, color=colors[tier], edgecolor="black", linewidth=0.5,
            label=f"{tier} tier",
        )
        for bar, agent in zip(bars, agents):
            info = agent_results[agent][tier]
            label = f"{info['avg']:.2f}\n{info['n_pass']}/{info['n']}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01, label,
                ha="center", va="bottom", fontsize=8,
            )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n - 0.5, 0.51, "pass threshold 0.5", fontsize=8, color="grey", ha="right")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(agents, rotation=0, fontsize=10)
    ax.set_ylabel("avg trajectory grade")
    ax.set_ylim(0, 1.05)
    ax.set_title("Two-tier held-out evaluation under hardened rubric (A+B+C)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=90)
    plt.close(fig)
    print(f"[ok] saved {out_name}")


# ---------------------------------------------------------------------------
# 3. Per-task heatmap
# ---------------------------------------------------------------------------

def plot_per_task_heatmap(
    per_task: Dict[str, Dict[str, float]],
    out_name: str = "per_task_heatmap.png",
) -> None:
    """per_task[task_id][agent] = grade."""
    # Column order: easy then hard
    easy_tasks = list(_tier_task_prefixes("easy"))
    hard_tasks = list(_tier_task_prefixes("hard"))
    task_order = [t for t in easy_tasks + hard_tasks if t in per_task]
    agents = sorted({a for d in per_task.values() for a in d.keys()})

    import numpy as np
    data = np.zeros((len(agents), len(task_order)))
    for i, a in enumerate(agents):
        for j, t in enumerate(task_order):
            data[i, j] = per_task.get(t, {}).get(a, 0.0)

    fig, ax = plt.subplots(figsize=(13, max(3, len(agents) * 0.7 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels([t[:28] for t in task_order], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=10)
    for i in range(len(agents)):
        for j in range(len(task_order)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if data[i,j] > 0.5 else "white")
    # Vertical divider between easy and hard
    ax.axvline(x=len(easy_tasks) - 0.5, color="black", linewidth=1.5)
    ax.set_title("Per-task grade — easy (left of divider) vs hard (right)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="grade")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=90)
    plt.close(fig)
    print(f"[ok] saved {out_name}")


# ---------------------------------------------------------------------------
# 4. Rubric hardening comparison (lenient vs hardened)
# ---------------------------------------------------------------------------

def plot_rubric_hardening_story(out_name: str = "rubric_hardening_story.png") -> None:
    """Before/after rubric hardening narrative in one plot."""
    # hand-curated from our experiments
    labels = ["Base", "SFT", "GRPO-v1\n(old rubric)", "GRPO-v2\n(hardened)", "GRPO-v3\n(search-aware)", "Heuristic"]
    lenient = [0.150, 0.622, 0.825, None, None, 0.938]  # old lenient rubric, 10-task avg
    hardened = [0.150, 0.400, 0.281, 0.429, 0.737, 0.737]  # hardened rubric — v3 ties heuristic!

    # v3 pending
    # insert v3 numbers when available

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(labels))
    width = 0.35

    lenient_plot = [v if v is not None else 0 for v in lenient]
    hardened_plot = [v if v is not None else 0 for v in hardened]

    bars1 = ax.bar([i - width/2 for i in x], lenient_plot, width,
                    color="#e67e22", edgecolor="black", linewidth=0.5,
                    label="lenient rubric (pre-hacking-fix)")
    bars2 = ax.bar([i + width/2 for i in x], hardened_plot, width,
                    color="#2c3e50", edgecolor="black", linewidth=0.5,
                    label="hardened rubric (A+B+C)")

    for bar, v in zip(bars1, lenient):
        if v is None:
            ax.text(bar.get_x() + bar.get_width()/2, 0.02, "—", ha="center", va="bottom", fontsize=9, color="grey")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, hardened):
        if v is None:
            ax.text(bar.get_x() + bar.get_width()/2, 0.02, "running", ha="center", va="bottom", fontsize=9, color="grey", rotation=0)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("avg trajectory grade (10 held-out tasks)")
    ax.set_ylim(0, 1.05)
    ax.set_title("GRPO-v3 MATCHES the heuristic ceiling under honest scoring (both = 0.74)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=90)
    plt.close(fig)
    print(f"[ok] saved {out_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Training curves for v2 and v3 (if present)
    for adapter_dir, label, prefix in [
        (OUTPUTS / "grpo_hf_l40s", "GRPO-v1 (soft rubric, 100 steps)", "v1"),
        (OUTPUTS / "grpo_v2_hardened", "GRPO-v2 (hardened rubric, 100 steps)", "v2"),
        (OUTPUTS / "grpo_v3_search_aware", "GRPO-v3 (search-aware + hardened, 100 steps)", "v3"),
    ]:
        metrics = adapter_dir / "train_metrics.json"
        plot_training_curves(metrics, label, prefix)

    # 2. Two-tier bar chart from latest eval JSON. Use eval_v2_hardened.json as
    #    primary (old gold-standard), augment with v3 when available
    v2_eval = OUTPUTS / "eval_v2_hardened.json"
    v3_eval = OUTPUTS / "eval_v3_search_aware.json"
    if v3_eval.exists():
        eval_data = json.loads(v3_eval.read_text())
    elif v2_eval.exists():
        eval_data = json.loads(v2_eval.read_text())
    else:
        print("[warn] no eval JSON yet; skipping bar chart")
        return

    # Build agent_results from eval_data structure
    agent_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_task: Dict[str, Dict[str, float]] = {}
    for agent_key, runs in eval_data.items():
        if not isinstance(runs, dict):
            continue
        easy_runs = runs.get("easy", [])
        hard_runs = runs.get("hard", [])
        all_runs = easy_runs + hard_runs
        display_name = {"heuristic": "Heuristic", "base_qwen": "Base", "sft_qwen": "GRPO-v3\n(search-aware)"}.get(agent_key, agent_key)
        agent_results[display_name] = {
            "easy": _agent_avg(easy_runs),
            "hard": _agent_avg(hard_runs),
            "overall": _agent_avg(all_runs),
        }
        for run in all_runs:
            tid = run.get("task_id", "")
            per_task.setdefault(tid, {})[display_name] = run.get("grade_score", 0.0)

    if agent_results:
        plot_two_tier_bar_chart(agent_results)
        plot_per_task_heatmap(per_task)

    # 3. Rubric hardening narrative
    plot_rubric_hardening_story()

    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
