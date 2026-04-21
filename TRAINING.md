# Training Guide — Spaces Pipeline Pro

End-to-end training workflow: from raw env to a GRPO-trained agent that orchestrates real HuggingFace Spaces.

---

## TL;DR

```bash
# 1. Generate gold trajectories from HeuristicAgent (free, fast)
python scripts/generate_gold_trajectories.py

# 2. (Optional) Distill trajectories from a frontier LLM (better quality, ~$5)
export OPENAI_API_KEY=...
python scripts/generate_llm_trajectories.py --tasks all --min-grade 0.5

# 3. SFT warmstart — teach the model tool-use JSON format
python scripts/sft_warmstart.py --output-dir outputs/sft_warmstart

# 4. GRPO — RL fine-tune on top of warmstart
python scripts/train_grpo.py --phase 1 --warmstart outputs/sft_warmstart --output-dir outputs/grpo_phase1
python scripts/train_grpo.py --phase 2 --warmstart outputs/grpo_phase1 --output-dir outputs/grpo_phase2
python scripts/train_grpo.py --phase 3 --warmstart outputs/grpo_phase2 --output-dir outputs/grpo_phase3
python scripts/train_grpo.py --phase 4 --warmstart outputs/grpo_phase3 --output-dir outputs/grpo_phase4

# 5. Evaluate on held-out
python scripts/evaluate.py --agent trained --model-path outputs/grpo_phase4
```

---

## The Gold Set

**What it is:** 63 high-quality agent trajectories (348 step-level pairs) covering 21 training tasks × 3 seeds each, all scoring grade ≥ 0.5.

**Where it comes from:**
| Source | Command | Cost | Output |
|---|---|---|---|
| HeuristicAgent (follows `gold_pipeline`) | `generate_gold_trajectories.py` | Free | `fixtures/sft_pairs.jsonl` (348 pairs) |
| LLM solver (GPT-5, Claude, Qwen 72B) | `generate_llm_trajectories.py` | ~$0.01/traj | `fixtures/llm_sft_pairs.jsonl` (append-only) |
| Self-play (trained model keeps best) | *not yet built* | Compute | After initial training |
| Hand-written | N/A | Time | Ad-hoc |

**The HeuristicAgent's gold set is enough to start.** It produces:
- Prompt: exactly what the LLM will see at each step
- Completion: the correct JSON action
- Reward: per-step shaping reward from the env

### Inspecting the data

```bash
head -1 fixtures/sft_pairs.jsonl | python -m json.tool
```

```json
{
  "task_id": "real_demo_audio_to_speech",
  "seed": 42,
  "step": 1,
  "prompt": "## Task: REAL DEMO: Take a famous speech audio clip...",
  "completion": "{\"action_type\": \"read_card\", \"payload\": {\"space_id\": \"hf-audio/whisper-large-v3\"}}",
  "reward": 0.1,
  "grade_score": 0.949
}
```

---

## Task Split

**21 training tasks** + **5 held-out eval tasks**:

Held-out (never seen during training):
- `multimodal_caption_speak_024`
- `multimodal_full_pipeline_025`
- `code_to_speech_020`
- `doc_quick_summary_015`
- `audio_sentiment_005`

This is deliberate — the held-out set tests **generalization** across:
- Unseen Space combinations
- Unseen domains (multimodal, long-horizon)
- New persona distributions

---

## Training Stages

### Stage 1: Generate Trajectories

```bash
# Heuristic (always do this first)
python scripts/generate_gold_trajectories.py --seeds 3

# LLM trajectories for diversity (optional)
export OPENAI_API_KEY=hf_xxx
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Start env server
uvicorn server.app:app --port 8000 &

# Collect
python scripts/generate_llm_trajectories.py \
    --tasks all \
    --episodes 3 \
    --min-grade 0.5 \
    --env-url http://localhost:8000
```

Outputs:
- `fixtures/gold_trajectories.jsonl` — full episodes (for analysis)
- `fixtures/sft_pairs.jsonl` — flattened prompt/completion (for SFT)
- `fixtures/grpo_prompts.jsonl` — task-level prompts (for GRPO)
- `fixtures/llm_sft_pairs.jsonl` — same but from LLM (optional)

### Stage 2: SFT Warmstart

Teaches the base model to output valid JSON actions. Without this, GRPO has to learn from scratch (slow, unstable).

```bash
python scripts/sft_warmstart.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --output-dir outputs/sft_warmstart
```

**Expected outcomes:**
- Validation loss: 2.5 → 0.3 over 3 epochs
- Model outputs valid JSON 90%+ of the time post-SFT (vs ~30% for base)
- ~15-30 min on free Colab T4

### Stage 3: GRPO with Curriculum

Four phases, each starting from the previous checkpoint:

| Phase | Tasks | Drift | Shaping | Generations | Steps |
|---|---|---|---|---|---|
| **1 Warmup** | Easy only | ❌ | Dense | 4 | 500 |
| **2 Discovery** | +Medium | ❌ | Dense (×0.7) | 6 | 500 |
| **3 Drift** | +Hard + schema drift | ✅ | Dense (×0.4) | 6 | 1000 |
| **4 Adversarial** | All + persona shifts | ✅ | Sparse | 8 | 1000 |

```bash
python scripts/train_grpo.py --phase 1 --warmstart outputs/sft_warmstart --output-dir outputs/grpo_p1
python scripts/train_grpo.py --phase 2 --warmstart outputs/grpo_p1 --output-dir outputs/grpo_p2
python scripts/train_grpo.py --phase 3 --warmstart outputs/grpo_p2 --output-dir outputs/grpo_p3
python scripts/train_grpo.py --phase 4 --warmstart outputs/grpo_p3 --output-dir outputs/grpo_p4
```

**Expected training curves:**

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Held-out |
|---|---|---|---|---|---|
| Final reward | 0.3 → 0.7 | 0.5 → 0.75 | 0.4 → 0.65 | 0.5 → 0.7 | 0.65 |
| Auditor flags / episode | 6 → 2 | 3 → 1.5 | 4 → 2 | 2 → 1 | 1.8 |
| Avg actions to submit | 15 → 9 | 11 → 8 | 13 → 9 | 10 → 7 | 8 |
| Schema drift recovery | — | — | 25% → 60% | 55% → 75% | 70% |

### Stage 4: Evaluation

```bash
python scripts/evaluate.py --agent trained --model-path outputs/grpo_p4 --output eval.json
```

Compares:
- Trained model: target ~0.65 on held-out
- HeuristicAgent: upper bound ~0.94 (it sees gold_pipeline)
- Random action baseline: ~0.2
- Untrained Qwen 1.5B: ~0.3

---

## Where More Data Comes From (When You Need More)

1. **Seed augmentation**: `--seeds 10` for 10× more trajectories (same distribution)
2. **LLM distillation**: Frontier models produce diverse reasoning paths
3. **Self-play**: After phase 1 GRPO, run trained model, filter by reward, add to SFT set
4. **Cross-persona**: Each trajectory varies by Expert persona — include all 3 for robustness
5. **Live traces**: Switch env to `SPACES_MODE=record` and let humans or strong LLMs run real tasks; record trajectories

---

## Hardware Requirements

| Setup | Model | Training time per phase | Notes |
|---|---|---|---|
| Colab T4 (free) | Qwen 1.5B + 4-bit + LoRA | 30-60 min | Use Unsloth |
| Colab Pro A100 | Qwen 3B + LoRA | 15-30 min | Recommended for hackathon |
| Local (consumer GPU) | Qwen 1.5B + LoRA | 20-40 min | Needs 12GB+ VRAM |
| H100 | Qwen 7B + LoRA | 10-20 min | Max performance |

For the hackathon venue (with sponsor GPU credits), Qwen 3B + LoRA on A100 is the sweet spot.

---

## Common Issues

**"Loss not decreasing during SFT"**
- Check tokenizer's chat template supports system+user+assistant
- Verify prompt doesn't exceed `--max-seq-length`
- Try lower LR (5e-5 instead of 2e-4)

**"GRPO reward curve flat at ~0"**
- SFT warmstart probably failed → verify model outputs JSON before GRPO
- `--num-generations` too low → increase to 6-8
- Reward function too sparse → use `--phase 1` first

**"Out of memory"**
- Reduce `--batch-size 1 --grad-accum 8`
- Reduce `--max-seq-length 2048`
- Use Unsloth 4-bit loading

**"gradio_client errors during training"**
- Training MUST run in `SPACES_MODE=mock` (the default)
- Never train with `live` or `hybrid` — rate limits + non-determinism kill training

---

## Files Produced

```
outputs/
├── sft_warmstart/             # LoRA checkpoint after SFT
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── grpo_p1/                   # Phase 1 checkpoint
├── grpo_p2/
├── grpo_p3/
└── grpo_p4/                   # Final trained model (use for eval)

fixtures/
├── gold_trajectories.jsonl    # Full episodes from HeuristicAgent
├── sft_pairs.jsonl            # Flattened for SFT (348 pairs)
├── llm_sft_pairs.jsonl        # From LLM solver (optional)
└── grpo_prompts.jsonl         # Task-level prompts for GRPO
```

---

## Quick Diagnostic Commands

```bash
# How many trajectories do I have?
wc -l fixtures/*.jsonl

# Average grade of gold trajectories
python -c "import json; trajs = [json.loads(l) for l in open('fixtures/gold_trajectories.jsonl')]; print(sum(t['grade_score'] for t in trajs) / len(trajs))"

# Verify SFT setup without installing heavy deps
python scripts/sft_warmstart.py --dry-run

# Verify GRPO setup
python scripts/train_grpo.py --dry-run --phase 1
```
