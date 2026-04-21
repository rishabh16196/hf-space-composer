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
  - multi-agent
  - tool-use
  - huggingface
---

# Spaces Pipeline Pro

**An RL training environment where agents learn to discover, compose, and adapt to live HuggingFace Spaces under multi-actor oversight.**

Built for the Meta PyTorch OpenEnv Hackathon, Round 2. Stacks **4 sub-theme bonuses**: Halluminate (multi-actor) + Fleet AI (oversight) + Patronus (schema drift) + Snorkel (evolving experts).

---

## The Problem

HuggingFace Hub hosts 500K+ Spaces — the world's largest catalog of AI tools. Yet:

- Frontier models (GPT-5, Claude 4.5) score **only 33-44%** on multi-tool orchestration benchmarks like GAIA
- There is **no RL training environment** built around Spaces as a multi-actor toolkit
- Real-world tool use breaks when API contracts drift, oversight checks fire, or success criteria evolve

**Spaces Pipeline Pro** trains agents to fluently use the entire HuggingFace ecosystem under realistic adversarial conditions.

---

## What Makes It Unique

We surveyed the existing landscape:

| Existing | What it covers | What's missing |
|---|---|---|
| **smolagents** | Framework for calling Spaces as tools | No training env, no oversight, no drift |
| **Calendar Gym** (in OpenEnv) | Real tool calls, multi-step workflows | Single tool surface, no multi-actor |
| **GAIA Benchmark** | Tests multi-tool reasoning | Static eval, no training |
| **ToolBench** | 16K APIs as fine-tuning data | Supervised, no interactive env |
| **BrowserGym** | Browser as RL env | Browser-only, no Hub orchestration |

**No one has built an RL training environment where the agent learns to discover, compose, and adapt to live HuggingFace Spaces under multi-actor oversight with schema drift.** That's the gap this env fills.

---

## Architecture at a Glance

```
                    ┌──────────────────────┐
                    │   Learning Agent     │
                    │  (Qwen 2.5 1.5B)     │
                    └─────────┬────────────┘
                              │ JSON action
                ┌─────────────┴──────────────┐
                │   Spaces Pipeline Env      │
                │   (OpenEnv-compatible)     │
                └──┬───────────┬─────┬────┬──┘
                   │           │     │    │
              ┌────▼───┐  ┌────▼───┐ │ ┌──▼──────────┐
              │ Spaces │  │Auditor │ │ │ Expert      │
              │ (mock  │  │ (Fleet │ │ │ Reviewer    │
              │  /live)│  │  AI)   │ │ │ (Snorkel)   │
              └────────┘  └────────┘ │ └─────────────┘
                                     │
                              ┌──────▼─────────┐
                              │  Schema Drift  │
                              │  (Patronus)    │
                              └────────────────┘
```

### Core mechanics

- **Multi-actor (Halluminate)**: Agent orchestrates 5-50 Spaces per task. Spaces are black-box actors with documented contracts.
- **Oversight Auditor (Fleet AI)**: Watches every action, raises flags for invalid inputs, decoy Spaces, redundant calls, hallucinations.
- **Schema Drift (Patronus)**: Mid-episode, Space contracts can change — field renames, type changes, deprecations. Agent must detect and adapt.
- **Evolving Expert Reviewer (Snorkel)**: Three personas (speed-first, accuracy-first, cost-first) score submissions. Personas can shift mid-episode.

---

## Quick Start

```bash
# Install
git clone <this-repo>
cd spaces_pipeline_env
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# Generate fixtures (one-time, synthetic)
python scripts/generate_fixtures.py

# Run server
uvicorn server.app:app --port 8000

# Run heuristic agent against 5 default tasks
python inference.py
```

---

## Live Demo

Use the demo runner with rich step-by-step output:

```bash
# Mock mode (instant, deterministic)
python scripts/demo_live.py --task multimodal_caption_speak_024 --agent hybrid

# Live mode (real HF Spaces fired)
SPACES_MODE=live HF_TOKEN=hf_xxx python scripts/demo_live.py --task audio_summarize_hindi_001 --agent hybrid
```

---

## Action Space

The agent has 4 discrete action types:

```python
{"action_type": "search_spaces", "payload": {"query": "audio transcription", "top_k": 5}}
{"action_type": "read_card",     "payload": {"space_id": "openai/whisper-large-v3"}}
{"action_type": "call_space",    "payload": {"space_id": "openai/whisper-large-v3",
                                              "inputs": {"audio_url": "...", "language": "hi"}}}
{"action_type": "submit",        "payload": {"answer": {"transcript": "...", "summary": "..."}}}
```

---

## Tasks

**25 task scenarios across 5 domains:**

| Domain | Tasks | Example |
|---|---|---|
| Audio | 5 | "Transcribe Hindi audio, summarize in English" |
| Vision | 5 | "Caption an image, translate to French" |
| Document | 5 | "Extract PDF text, identify entities" |
| Code | 5 | "Explain code, translate explanation to Hindi" |
| Multimodal | 5 | "From audio + image, produce a unified summary" |

20 for training, 5 held-out for evaluation.

Each task has:
- Natural-language description
- Expected output schema
- Action and Space-call budgets
- Expert persona + rubric
- Optional drift events with trigger steps

---

## Reward Design

```
final_reward = (
    0.50 * expert_score        # Snorkel-rated quality
  + 0.20 * auditor_score       # 1 - severity-weighted flag penalty
  + 0.15 * efficiency_score    # 1 - actions_used / budget
  + 0.10 * cost_score          # 1 - spaces_called / budget
  + 0.05 * format_score        # output schema completeness
)
```

Plus optional **dense intermediate rewards** during warmup:
- +0.05 for valid search
- +0.10 for reading card
- +0.20 for successful Space call
- -0.20 for failed call
- -0.10 for redundant call
- Auditor flag penalties: -0.05 (warning) / -0.15 (error) / -0.40 (critical)

Dense rewards are annealed to zero across training phases.

---

## Training

**Stack:** Unsloth + TRL GRPO + Qwen 2.5 1.5B-Instruct + LoRA

**Curriculum (4 phases):**

| Phase | Tasks | Drift | Personas | Reward shaping | Steps |
|---|---|---|---|---|---|
| 1 Warmup | Easy only | ❌ | Speed-first | Dense | 5K |
| 2 Discovery | Easy + Medium | ❌ | All 3 | Dense (×0.7) | 5K |
| 3 Drift | All | ✅ | All 3 | Dense (×0.4) | 10K |
| 4 Adversarial | All | ✅ | Mid-episode shifts | Sparse | 10K |

Run training (at venue with sponsor compute):

```bash
python scripts/train_grpo.py --phase 1 --steps 5000 --output-dir outputs/phase1
python scripts/train_grpo.py --phase 2 --steps 5000 --output-dir outputs/phase2
python scripts/train_grpo.py --phase 3 --steps 10000 --output-dir outputs/phase3
python scripts/train_grpo.py --phase 4 --steps 10000 --output-dir outputs/phase4
```

End-to-end Colab notebook: [`colab/train_spaces_pipeline.ipynb`](colab/train_spaces_pipeline.ipynb)

---

## Evaluation

```bash
# Baseline heuristic
python scripts/evaluate.py --agent heuristic

# Trained agent
python scripts/evaluate.py --agent trained --model-path outputs/phase4

# LLM zero-shot (frontier baseline)
OPENAI_API_KEY=... python scripts/evaluate.py --agent llm --model gpt-4o-mini
```

---

## Modes

| Mode | Speed | Use case | Set via |
|---|---|---|---|
| `mock` (default) | ~30 ms / call | Training, dev, reproducibility | `SPACES_MODE=mock` |
| `live` | 2-30 s / call | Demos, final eval | `SPACES_MODE=live` (needs `HF_TOKEN`) |
| `record` | 2-30 s / call | Refresh fixtures from real Spaces | `SPACES_MODE=record` |
| `hybrid` | 30 ms cached / 2-30 s on miss | Iterative dev + demo safety net. Cache hits are fast; misses auto-fetch live and persist for future hits. | `SPACES_MODE=hybrid` |

**Hybrid mode** cascade:
- `search`: local catalog first; supplement with live results and auto-append new Spaces
- `read_card`: cached card first; miss → live fetch → cache
- `call_space`: preferred order is real recording → exact hash → any cached → live (cached with `_real: true` marker)

---

## Sub-Theme Coverage

| Sub-theme | Sponsor | How we hit it |
|---|---|---|
| **Halluminate** | Multi-Actor Environments | Spaces are black-box actors; agent must discover and orchestrate them |
| **Fleet AI** | Scalable Oversight | Auditor agent monitors every action, flags violations with severity |
| **Patronus** | Consumer Workflows w/ Schema Drift | Space contracts drift mid-episode; agent must detect and re-adapt |
| **Snorkel** | Simulated Experts in the Loop | Reviewer with 3 evolving personas scoring submissions |

Plus main themes:
- ✅ Theme #1 Multi-Agent
- ✅ Theme #3.1 World Modeling (real Hub as world)
- Bonus: arguably Theme #2 Long-Horizon (200+ step pipelines)

---

## Project Structure

```
spaces_pipeline_env/
├── inference.py                  # Hybrid (heuristic + LLM) agent
├── models.py                     # Action/Observation schemas
├── client.py                     # EnvClient (HTTP/WebSocket)
├── __init__.py
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml
├── server/
│   ├── app.py                    # FastAPI app
│   ├── spaces_pipeline_environment.py  # Core Environment
│   ├── auditor.py                # Fleet AI oversight
│   ├── expert_reviewer.py        # Snorkel evolving expert
│   ├── schema_drift.py           # Patronus drift mechanic
│   ├── space_catalog.py          # Hub search/cards (mock + live)
│   ├── space_caller.py           # Space invocation (mock + live)
│   ├── rubrics.py                # TrajectoryRubric grader
│   └── Dockerfile
├── fixtures/
│   ├── tasks.json                # 25 task definitions
│   ├── space_catalog.json        # Mock catalog
│   ├── cards/                    # Per-Space cards
│   └── responses/                # Per-task mock responses
├── scripts/
│   ├── generate_fixtures.py      # Auto-gen synthetic fixtures
│   ├── record_fixtures.py        # Record live HF responses
│   ├── train_grpo.py             # TRL + Unsloth training
│   ├── evaluate.py               # Held-out evaluation
│   └── demo_live.py              # Pretty-printed live demo
└── colab/
    └── train_spaces_pipeline.ipynb  # End-to-end Colab notebook
```

---

## Pitch Storyline (3 minutes)

1. **0:00-0:30** Hook: "500K HF Spaces, but frontier models score 33% on multi-tool orchestration. We trained an agent to use HuggingFace fluently."
2. **0:30-1:00** Show baseline (untrained Qwen 1.5B) failing on a task — wrong Space chosen, ignored Auditor flags, broken submission.
3. **1:00-1:30** Show training curves (4 metrics rising over 30K rollouts across 4 phases).
4. **1:30-2:30** **LIVE DEMO**: trained agent given a fresh task, orchestrates 3-5 real HF Spaces on screen, with Auditor commentary scrolling.
5. **2:30-2:50** Stress test: simulate Space deprecation, watch agent recover.
6. **2:50-3:00** Sub-theme coverage diagram (4 colored blocks lighting up).

---

## License

Apache-2.0
