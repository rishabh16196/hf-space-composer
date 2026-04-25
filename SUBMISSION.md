# Submission — Spaces Pipeline Pro

Single-page reference for: hackathon judges, follow-up sessions, anyone trying to reproduce.

---

## 1. The links (paste these into the submission form)

| What | URL |
|---|---|
| **Environment on HF Spaces** (judge entry point) | https://huggingface.co/spaces/rishabh16196/spaces-pipeline-pro |
| GitHub source | https://github.com/rishabh16196/hf-space-composer |
| Colab training notebook | https://github.com/rishabh16196/hf-space-composer/blob/main/colab/train_spaces_pipeline.ipynb |
| Final GRPO adapter (v3) | https://huggingface.co/rishabh16196/spaces-pipeline-grpo-1.5b-v3-search-aware |
| SFT adapter (search-aware) | https://huggingface.co/rishabh16196/spaces-pipeline-sft-1.5b-search-aware |

The other 3 adapters (v1 lenient, v2 hardened, original SFT) are public on the same HF account if a judge wants to reproduce the rubric-hardening arc.

---

## 2. Minimum requirements check

| Requirement | Status | Where |
|---|---|---|
| OpenEnv (latest release) | ✅ | `pyproject.toml` pins `openenv-core[core]>=0.2.2` |
| Training script using Unsloth or TRL | ✅ | `local_training/grpo_unsloth.py` (Unsloth + custom trajectory-level GRPO) |
| Colab notebook | ✅ | `colab/train_spaces_pipeline.ipynb` |
| Evidence of training (loss + reward plots) | ✅ | `local_training/outputs/plots/` — 9 PNGs incl. reward + KL curves |
| Mini-blog OR <2-min video OR slide deck | ⚠️ user-side | see Section 6 below |
| HF Space hosting | ✅ | live at the URL above |
| README motivating the problem + results | ✅ | `README.md` with embedded plots + 6-agent results table |

---

## 3. The headline result

**Two-tier held-out, 10 tasks, hardened rubric**:

|   | Base | **GRPO-v3** | Heuristic (gold-pipeline ceiling) |
|---|---|---|---|
| EASY (5) | 0.150 · 0/5 | 0.813 · 4/5 | 0.922 · 5/5 |
| **HARD (5)** | 0.150 · 0/5 | **0.659 · 3/5** 🚀 | 0.552 · 2/5 |
| **ALL (10)** | 0.150 | **0.737 · 7/10** | **0.737 · 7/10** |

**GRPO-v3 ties the heuristic on overall and beats it on the HARD tier** (the long-horizon + 50-step-marathon-with-drift tasks). Story arc: 4 training iterations, with reward hacking caught and rubric-hardened mid-run.

---

## 4. How to run training (HF Jobs, ~$3, ~1.5 hours)

The entrypoint script lives in the repo at `scripts/hf_jobs/train_sft_grpo.sh`. It:
1. Installs torch 2.8+ + Unsloth via the meta-pytorch/OpenEnv recipe (`uv pip` + `--no-deps` second pass)
2. Clones the repo at HEAD
3. Verifies the hardened rubric and search-aware data are loaded (assertion fails fast)
4. Runs SFT (3 epochs on 651 search-aware pairs) — ~4 min on L40S
5. Runs multi-step GRPO (100 steps, B=4, G=8, hardened rubric) — ~80 min
6. Uploads both adapters back to HF Hub

### Launch command (from any machine with `hf` CLI authed):

```bash
hf jobs run \
  --flavor l40sx1 \
  --timeout 3h \
  --secrets HF_TOKEN \
  --detach \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  bash -c "$(cat scripts/hf_jobs/train_sft_grpo.sh)"
```

### Why these specific choices
| Choice | Why |
|---|---|
| **base image** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` | smallest image with working CUDA 12.4. Torch is upgraded in-script. |
| **flavor** `l40sx1` ($1.80/hr, 48 GB VRAM) | enough for Qwen 1.5B + Unsloth + KV cache. **Avoid h200** — confirmed CUDA `Error 802: system not yet initialized` with cu124 torch builds. **Avoid a100-large queue** — was ~30 min vs L40S's 2-5 min during our runs. |
| **torch ≥ 2.8.0** | Unsloth latest needs ≥2.8 (we tried 2.5.1, doesn't work) |
| **`uv pip --no-deps` second pass** | locks `transformers==4.56.2`, `trl==0.22.2`, `unsloth`, `unsloth_zoo` versions so the resolver doesn't pull torch 2.10 nightly |
| **`build-essential` in apt** | Triton JIT needs `gcc`. Without this you get `Failed to find C compiler`. |
| **`--no-deps` on Unsloth git install** | prevents flash-attn from being source-built (it'd need torch at build time, which pip's build-isolation hides from it) |

### Common failure modes (in case the next attempt hits one)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` during flash-attn build | flash-attn pulled in transitively; use `--no-deps` on Unsloth install |
| `ImportError: cannot import name 'FSDPModule'` | TRL version ≥ 0.13 wants torch 2.6+; pin `trl==0.22.2` (which is pre-FSDPModule reliance) |
| `Unsloth: torch==2.10.0 requires torchvision>=0.25.0` | upgrade torchvision alongside torch (`"torch>=2.8.0" "torchvision>=0.25.0"`) |
| `cuda: False` despite GPU visible to nvidia-smi | hardware/driver/torch-CUDA mismatch — common on H200 with cu124 torch. Switch to L40S. |
| `Failed to find C compiler` | `apt install -y build-essential` |
| `_pickle.UnpicklingError: invalid load key` when loading SFT adapter | use `safetensors.torch.load_file()`, not `torch.load()`, on `.safetensors` files |
| `SFT weights loaded: 0, skipped: 392` | Unsloth's PEFT key naming differs from plain PEFT. Use `FastLanguageModel.from_pretrained(model_name=adapter_path)` to let Unsloth auto-load — don't manual-copy weights. |

All of the above are baked into `scripts/hf_jobs/train_sft_grpo.sh` already.

---

## 5. How to run eval-only (sanity reproduction, ~$0.20)

For a judge who wants to verify the headline numbers without retraining:

```bash
hf jobs run \
  --flavor l40sx1 \
  --timeout 30m \
  --secrets HF_TOKEN \
  --detach \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  bash -c "$(cat scripts/hf_jobs/eval_only.sh)"
```

Defaults to evaluating `rishabh16196/spaces-pipeline-grpo-1.5b-v3-search-aware`. Override with `ADAPTER_REPO=...` env var. Expected output ends with:
```
ALL(10)    avg=0.150 pass=0/10     avg=0.737 pass=7/10     avg=0.737 pass=7/10
```

---

## 6. How to run locally (no GPU needed, MPS or CPU)

```bash
git clone https://github.com/rishabh16196/hf-space-composer.git
cd hf-space-composer
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# Smoke test the env (HeuristicAgent on 1 task, ~5 sec)
python inference.py

# Run a full agent comparison (all 4 agents on 5 hard tasks, ~15 min on MPS)
cd local_training
python eval_two_tier.py
```

---

## 7. Repo layout (what's where)

```
hf-space-composer/
├── README.md                          # ← judge entry point (links + headline + plots)
├── SUBMISSION.md                      # ← this file (operations doc)
├── PITCH.md                           # original Round-1 pitch
├── REWARD_HACKING_FIX_PLAN.md         # rubric design analysis
├── BLOG_POST_DRAFT.md                 # narrative reference (NOT published)
├── Dockerfile                         # HF Space deployment (uvicorn server.app:app)
├── pyproject.toml                     # openenv-core + gradio_client + huggingface_hub
├── openenv.yaml                       # OpenEnv manifest
├── inference.py                       # 3 agents: HeuristicAgent (search-aware), LLMAgent, HybridAgent
├── models.py                          # Action / Observation pydantic schemas
├── client.py                          # OpenEnv HTTP client wrapper
├── server/
│   ├── app.py                         # FastAPI app (OpenEnv contract)
│   ├── spaces_pipeline_environment.py # core Environment (reset/step/state)
│   ├── auditor.py                     # rule-based oversight
│   ├── expert_reviewer.py             # expert persona + per-task rubric
│   ├── schema_drift.py                # mid-episode drift injection
│   ├── space_catalog.py               # HF Hub search/cards (mock + live)
│   ├── space_caller.py                # gradio_client invocation
│   ├── rubrics.py                     # TrajectoryRubric grader (Fix A+B+C hardened)
│   └── Dockerfile                     # OpenEnv-base wrapper variant
├── fixtures/
│   ├── tasks.json                     # 38 tasks across 5 domains + 5 long + 2 marathon
│   ├── space_catalog.json             # 5,002 real HF Spaces (scraped)
│   ├── cards/                         # per-Space card cache
│   ├── responses/                     # per-task mock response cache
│   ├── sft_pairs.jsonl                # 651 SFT pairs (32.7% search actions)
│   └── gold_trajectories.jsonl        # full episode-level gold runs
├── scripts/
│   ├── hf_jobs/
│   │   ├── train_sft_grpo.sh          # HF Jobs entrypoint: SFT + GRPO + upload
│   │   └── eval_only.sh               # HF Jobs entrypoint: download + eval
│   ├── train_grpo.py                  # CUDA + Unsloth GRPO (alt path)
│   ├── generate_gold_trajectories.py  # heuristic → SFT pairs generator
│   ├── evaluate.py                    # held-out eval runner
│   ├── demo_live.py                   # pretty per-step demo
│   └── ... (scrapers, fixture tools)
├── local_training/
│   ├── sft_local.py                   # plain transformers+peft SFT (MPS/CUDA)
│   ├── grpo_unsloth.py                # multi-step GRPO with Unsloth (CUDA only)
│   ├── grpo_multistep.py              # multi-step GRPO with plain PEFT (MPS fallback)
│   ├── eval_two_tier.py               # easy + hard held-out evaluation
│   ├── make_plots.py                  # PNG generator
│   ├── trace_long_tasks.py            # full per-step trajectory dumps
│   ├── SLIDE_STORYLINE.md             # internal pitch narrative
│   └── outputs/
│       ├── plots/                     # 9 PNGs
│       └── (adapter dirs, eval logs)
└── colab/
    └── train_spaces_pipeline.ipynb    # judge-runnable Colab
```

---

## 8. What to record for the video / slides (if you make one)

Tight 90-second narration script:

> **(0–10s)** "HuggingFace Hub has 500,000+ Spaces. We built an OpenEnv environment that trains LLM agents to discover, compose, and recover from drift across 5,002 real ones."
>
> **(10–25s)** "The agent does `search_spaces`, `read_card`, `call_space`, `submit`. Tasks span audio, vision, docs, code, multimodal — including 50-step marathons where Space APIs drift mid-episode."
>
> **(25–45s)** *Show `rubric_hardening_story.png`* "Our first GRPO run hit 0.83 on the hard tier. Per-step traces revealed it was reward hacking — submitting placeholder answers with 4 calls instead of 20."
>
> **(45–65s)** *Show `two_tier_bar_chart.png`* "We hardened the rubric three ways, regenerated training data with search-aware demonstrations, retrained. GRPO-v3 ties the heuristic at 0.737 — and on the HARD tier with the marathons, GRPO-v3 beats it 0.66 to 0.55."
>
> **(65–85s)** "What this proves: the env's iterate-rubric-alongside-agent loop works. We caught our own reward hack via per-task tracing, fixed it, and got honest gains."
>
> **(85–90s)** "Code, training scripts, and adapters are all open at the GitHub link in the description."

Suggested shots:
1. README hero plot (`rubric_hardening_story.png`) — first 25s
2. Live trace of GRPO-v3 on `marathon_news_evolving_036` — 25-45s
3. Bar chart (`two_tier_bar_chart.png`) — 45-65s
4. GitHub repo browse + HF Space — 65-90s

---

## 9. Submission checklist

- [x] OpenEnv compliance (`openenv.yaml`, base classes, gym-style API)
- [x] Working training script (Unsloth + custom GRPO, runs in Colab)
- [x] Real training evidence (9 plot PNGs + train_metrics.json files)
- [x] HF Space hosting the env
- [x] README with motivation + results + plots embedded
- [ ] Mini-blog / video / slide deck — **user to produce**, link from README when ready
- [x] All assets pushed to both GitHub `main` and HF Space `main`

Run the submission form. Done.
