---
title: Spaces Pipeline Pro
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - long-horizon
  - tool-use
  - schema-drift
  - huggingface
---

# Spaces Pipeline Pro

**An OpenEnv-compatible RL training environment for long-horizon tool orchestration with live schema drift.** Agents learn to discover, compose, and recover-from-failure across 5,000+ real HuggingFace Spaces.

> **Hackathon theme mapping** — this env fits Round 2 **Theme #2 (Long-Horizon Planning & Instruction Following)** as the primary fit, and **Theme #3.1 (World Modeling / Professional Tasks — tool-discovery benchmark)** as a secondary fit.

---

## What makes it hard

The agent's job looks simple — "answer the question using the right HuggingFace Spaces" — but the environment enforces four pressures that no existing tool-use benchmark stacks together:

1. **Long horizons, sparse reward.** Marathon tasks (`marathon_news_evolving_036`, `marathon_investigation_037`) have 50-step action budgets. Grade is a single scalar at `submit`. The agent must decompose the goal, track state across 50+ observations, and **recover from early mistakes** (Theme #2 target behavior).
2. **Schema drift injected mid-episode.** Live Spaces rename fields, deprecate endpoints, or change types unannounced. The agent's prior plan stops working — it has to detect the drift, read a new card, and re-plan with the new API.
3. **Tool discovery in the action space.** 5,002 real Spaces indexed; the agent must `search_spaces` to find candidates, `read_card` to learn their schemas, then compose them. No fixed tool list.
4. **Multi-objective oversight.** An auditor monitors every action for rule violations; an expert reviewer scores the final answer under one of three hidden quality personas. Format, time, cost, and efficiency all contribute to the final grade.

---

## Why it's a fit for Theme #2 + #3.1

| Requirement from the brief | How this env satisfies it |
|---|---|
| *"deep, multi-step reasoning with sparse or delayed rewards"* | 50-step marathons; grade only at `submit` |
| *"decompose goals, track state over extended trajectories"* | Tasks chain 3–12 Spaces; prior outputs drive next step |
| *"recover from early mistakes"* | Schema drift forces re-planning after bad calls |
| *"long running sessions beyond context memory limits"* | Marathon observations push prompt length hard |
| *"real interaction with tools, APIs, or dynamic systems"* | `gradio_client` calls 5,002 real HF Spaces |
| *"Tool-discovery benchmarks"* | `search_spaces` + `read_card` are first-class actions |
| *"maintain consistent internal state, orchestrate multi-step workflows"* | Agent must remember which Spaces drifted, which failed |

---

## What's novel vs the existing landscape

| Existing work | What it covers | What's missing |
|---|---|---|
| **smolagents** | Framework for calling Spaces as tools | No training env, no drift, no oversight |
| **Calendar Gym** (OpenEnv) | Single-tool RL env | Single surface, no discovery problem |
| **GAIA Benchmark** | Multi-tool reasoning eval | Static, not trainable |
| **ToolBench** | 16K APIs, SFT data | Supervised-only, no interactive env |
| **BrowserGym** | Web as RL env | Browser-only, no Hub orchestration |

No prior work combines **(a) live tool discovery across the HF Spaces catalog + (b) mid-episode schema drift + (c) 50-step marathon horizons + (d) multi-component graded reward** into a single trainable env. That's the gap this fills.

---

## Quick Start

```bash
git clone https://github.com/rishabh16196/hf-space-composer
cd hf-space-composer
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# Smoke test (mock mode, ~2s)
python inference.py

# Rich step-by-step demo on a marathon task
python scripts/demo_live.py --task marathon_news_evolving_036 --agent heuristic
```

## OpenEnv HTTP server

```bash
uvicorn server.app:app --port 8000
# Observation endpoint: POST http://localhost:8000/reset
# Step endpoint:        POST http://localhost:8000/step
```

---

## Action Space

4 discrete action types — the agent emits JSON:

```python
{"action_type": "search_spaces", "payload": {"query": "audio transcription", "top_k": 5}}
{"action_type": "read_card",     "payload": {"space_id": "openai/whisper-large-v3"}}
{"action_type": "call_space",    "payload": {"space_id": "openai/whisper-large-v3",
                                              "inputs": {"audio_url": "...", "language": "hi"}}}
{"action_type": "submit",        "payload": {"answer": {"transcript": "...", "summary": "..."}}}
```

---

## Tasks (38 scenarios across 5 domains)

| Domain | Tasks | Example |
|---|---|---|
| Audio | 5 | "Transcribe Hindi audio, summarize in English" |
| Vision | 5 | "Caption an image, translate to French" |
| Document | 5 | "Extract PDF text, identify named entities" |
| Code | 5 | "Explain code, translate explanation to Hindi" |
| Multimodal | 11 | "From audio + image, produce a unified summary" |
| **Long-horizon (7–12 Space calls)** | 5 | Multi-domain chains, e.g. doc → translate → TTS |
| **Marathon (50-step budget, drift injected)** | 2 | Evolving news brief; investigation pipeline |

**Two-tier held-out split** for honest reporting:
- **Easy held-out** (5): mid-difficulty, 3–10 steps. Tests "did SFT produce a valid agent?"
- **Hard held-out** (5): 7+ step long-horizon plus both 50-step marathons. Tests "does training solve the hard problem?"

28 tasks for training. Each task carries:
- Natural-language description + expected output schema
- Action and Space-call budgets
- A hidden expert persona (speed_first / accuracy_first / cost_first) and a per-task rubric
- Optional schema-drift events with trigger steps

---

## Reward Design

```
final_grade = (
    0.45 * expert_score       # expert reviewer, persona-weighted
  + 0.18 * auditor_score      # 1 − severity-weighted flag penalty
  + 0.12 * efficiency_score   # actions used / budget
  + 0.12 * time_score         # wall-clock time / budget
  + 0.08 * cost_score         # space calls / budget
  + 0.05 * format_score       # output schema completeness
) normalized to [0, 1]
```

**Engagement gate** — caps lazy-submit reward-hacking:
```python
if task_requires_tools and successful_space_calls == 0:
    final_grade = min(final_grade, 0.15)
```
(We discovered this loophole empirically when base Qwen was scoring 0.46 by submitting an empty answer on step 1 — all budget components were 1.0 because no resources were spent. Fixed before GRPO was run.)

Optional **dense intermediate rewards** for training warmup (annealed to zero across phases): +0.05 search, +0.10 read_card, +0.20 successful call, −0.20 failure, −0.10 redundant, −0.05/−0.15/−0.40 for auditor warning/error/critical flags.

---

## Training

**Stack**: Unsloth + TRL + Qwen 2.5 1.5B-Instruct + LoRA r=16

### SFT warmstart (local, 10 min on Apple Silicon)

```bash
cd local_training
python sft_local.py                    # 474 pairs, 1 epoch → LoRA adapter
python eval_two_tier.py                # easy/hard held-out comparison
```

### Multi-step GRPO (venue compute — Unsloth + CUDA)

```bash
# Unsloth version — matches venue target; model acts at every env step
cd local_training
python grpo_unsloth.py \
  --adapter outputs/sft_local_1.5b_clean \
  --max-steps 100 --num-gens 6 --batch-size 4
```

### End-to-end Colab notebook

[`colab/train_spaces_pipeline.ipynb`](colab/train_spaces_pipeline.ipynb) — clone repo, install Unsloth stack, run SFT + GRPO + eval in one notebook.

### HF Jobs (tested pipeline)

```bash
hf jobs run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  bash -c "$(cat hf_job_entrypoint.sh)"
```

---

## Results (SFT + multi-step GRPO)

Two-tier held-out, 10 tasks × 4 agents:

| Tier | Base Qwen 1.5B | SFT 1.5B | **SFT + GRPO 1.5B** | HeuristicAgent (ceiling) |
|---|---|---|---|---|
| **EASY (5)** | 0.150 · 0/5 | 0.594 · 3/5 | **0.658 · 3/5** | 0.960 · 5/5 |
| **HARD (5)** | 0.150 · 0/5 | 0.650 · 3/5 | **0.992 · 5/5** 🚀 | 0.915 · 5/5 |
| **ALL (10)** | 0.150 · 0/10 | 0.622 · 6/10 | **0.825 · 8/10** | 0.938 · 10/10 |

**The big win is on the HARD tier**: GRPO took the agent from 3/5 at avg 0.65 → **5/5 at avg 0.99**, basically matching the gold-pipeline HeuristicAgent's ceiling. Three previously-failing tasks (`code_to_speech_020`, `long_doc_localize_032`, `long_meeting_analysis_034`) all jumped to 0.97-0.998.

GRPO config: **100 steps, Unsloth + L40S (HF Jobs), ~55 min, ~$1.80**. Multi-step trajectory rollouts (the model generates every action, no heuristic fallback). See `local_training/SLIDE_STORYLINE.md` for the full pitch narrative.

---

## Modes

| Mode | Speed | Use case | Set via |
|---|---|---|---|
| `mock` (default) | ~30 ms / call | Training, dev, reproducibility | `SPACES_MODE=mock` |
| `live` | 2–30 s / call | Demos, final eval | `SPACES_MODE=live` (needs `HF_TOKEN`) |
| `record` | 2–30 s / call | Refresh fixtures from real Spaces | `SPACES_MODE=record` |
| `hybrid` | 30 ms cached / 2–30 s on miss | Iterative dev + demo safety net | `SPACES_MODE=hybrid` |

---

## Project Structure

```
spaces_pipeline_env/
├── inference.py                  # Heuristic + LLM agents
├── models.py                     # Action/Observation pydantic schemas
├── client.py                     # OpenEnv HTTP client
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml
├── Dockerfile                    # HF Space + OpenEnv docker wrapper
├── server/
│   ├── app.py                    # FastAPI app (OpenEnv contract)
│   ├── spaces_pipeline_environment.py  # Core Environment
│   ├── auditor.py                # Rule-based oversight
│   ├── expert_reviewer.py        # Expert persona + rubric
│   ├── schema_drift.py           # Mid-episode drift injection
│   ├── space_catalog.py          # Hub search + cards (mock + live)
│   ├── space_caller.py           # Gradio-client invocation
│   └── rubrics.py                # TrajectoryRubric grader
├── fixtures/
│   ├── tasks.json                # 38 task definitions
│   ├── space_catalog.json        # 5,002 Spaces (scraped from HF)
│   ├── cards/                    # Per-Space card cache
│   └── responses/                # Per-task mock response cache
├── scripts/
│   ├── generate_gold_trajectories.py  # HeuristicAgent rollouts → SFT pairs
│   ├── train_grpo.py                  # Unsloth + CUDA GRPO (venue)
│   ├── evaluate.py                    # Held-out evaluation runner
│   └── demo_live.py                   # Pretty step-by-step demo
├── local_training/
│   ├── sft_local.py              # MPS-compatible SFT (Apple Silicon)
│   ├── grpo_unsloth.py           # Multi-step GRPO w/ Unsloth (CUDA)
│   ├── grpo_multistep.py         # Multi-step GRPO w/ plain PEFT (MPS fallback)
│   ├── eval_two_tier.py          # Easy + Hard held-out evaluation
│   └── SLIDE_STORYLINE.md        # Pitch narrative w/ numbers
└── colab/
    └── train_spaces_pipeline.ipynb
```

---

## License

Apache-2.0
