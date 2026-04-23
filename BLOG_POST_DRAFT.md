# Teaching an LLM to Survive When Its Tools Lie

> An OpenEnv environment for training agents on tool-discovery + schema-drift recovery over 5,002 real HuggingFace Spaces. Built for the Meta PyTorch OpenEnv Hackathon (Round 2).

**TL;DR** — We built an RL environment where an LLM agent has to orchestrate real HuggingFace Spaces (scraped 5,002 of them) to solve tasks spanning audio, vision, documents, code, and multimodal. The twist: Spaces drift mid-episode (APIs rename fields without warning), and the reward function is multi-component so the agent can't game it with a one-liner submit. We trained Qwen 2.5 1.5B with SFT + GRPO, caught our own reward hacking, hardened the rubric three ways, retrained, and report honest numbers.

**Links**
- Env on HF Spaces: [rishabh16196/spaces-pipeline-pro](https://huggingface.co/spaces/rishabh16196/spaces-pipeline-pro)
- GitHub: [rishabh16196/hf-space-composer](https://github.com/rishabh16196/hf-space-composer)
- Colab notebook: [`colab/train_spaces_pipeline.ipynb`](https://github.com/rishabh16196/hf-space-composer/blob/main/colab/train_spaces_pipeline.ipynb)
- Trained adapters: [SFT](https://huggingface.co/rishabh16196/spaces-pipeline-sft-1.5b-search-aware), [GRPO v3](https://huggingface.co/rishabh16196/spaces-pipeline-grpo-1.5b-v3-search-aware)

**Hackathon theme fit**: Theme #2 (Long-Horizon Planning) + Theme #3.1 (Tool Discovery / World Modeling).

---

## The problem — why train on this?

Frontier LLMs score only 33–44% on multi-tool benchmarks like GAIA. HuggingFace Hub has **500,000+ Spaces** — the largest public catalog of AI tools in the world — yet there's no training environment built around it. Existing benchmarks (GAIA, ToolBench, smolagents, BrowserGym) either test-only, synthetic-only, or single-surface.

Worse, every production tool-use agent faces three nasty realities that standard benchmarks ignore:

1. **Tool discovery** — you don't know which of 500K Spaces does the job until you search.
2. **Schema drift** — APIs change. The payload key named `text` might be renamed to `input` without warning.
3. **Reward hacking** — if your grading is a weighted sum, a clever agent will find the null-work sweet spot.

Our environment exposes all three, at scale.

---

## The environment at a glance

| Dimension | Value |
|---|---|
| Tool surface | **5,002 real HuggingFace Spaces** (scraped live from HF Hub) |
| Tasks | **38 scenarios** across 5 domains (audio, vision, doc, code, multimodal) |
| Long-horizon | 5 tasks at 7–12 steps |
| **Marathon** | **2 tasks at 50-step action budgets with schema drift injected mid-episode** |
| Action space | `search_spaces`, `read_card`, `call_space`, `submit` |
| Rubric | 6 components (expert, auditor, efficiency, cost, time, format) + engagement gate + value-diversity + grounding check |
| Base image | OpenEnv compatible (`openenv-core[core]>=0.2.2`) |

The agent sees a natural-language task description, an expected output schema, observations from recent actions, and search/card-read results. It has to compose a pipeline of Space calls that produces outputs matching the schema.

---

## What the agent actually does

On task `multimodal_caption_speak_024` ("Caption an image, then read the caption aloud"):

```
step 1  search_spaces {"query": "image caption"}       → top_k=5 candidates
step 2  read_card     {"space_id": "fancyfeast/joy-caption-pre-alpha"}
step 3  call_space    {"space_id": "fancyfeast/joy-caption-pre-alpha",
                       "inputs": {"input_image": "<url>"}}
step 4  search_spaces {"query": "text to speech"}       → TTS candidates
step 5  read_card     {"space_id": "innoai/Edge-TTS-Text-to-Speech"}
step 6  call_space    {"space_id": "innoai/Edge-TTS-Text-to-Speech",
                       "inputs": {"text": "<caption>", "voice": "en-US"}}
step 7  submit        {"answer": {"caption": "...", "audio_url": "..."}}
```

On `marathon_news_evolving_036` (50-step budget, drift event at step 10):

```
...
step 9   call_space pszemraj/summarize-long-text {"text": "..."}  → returns summary
step 10  [DRIFT EVENT] field "text" renamed to "input"; card re-issued
step 11  read_card   pszemraj/summarize-long-text  → learns new schema
step 12  call_space  pszemraj/summarize-long-text {"input": "..."}  → recovers
...
```

That drift-recovery pattern is what Theme #2 ("recover from early mistakes") is asking for.

---

## Training pipeline

- **Model**: Qwen 2.5 1.5B Instruct + LoRA r=16
- **SFT**: 3 epochs on 651 gold-trajectory pairs (from a search-aware HeuristicAgent baseline)
- **GRPO**: 100 steps of multi-step trajectory-level GRPO (model generates at every env step, not just step 0) with Unsloth on L40S via HF Jobs
- **Rubric**: hardened 6-component weighted sum + three post-hoc gates (see below)
- **Time/cost**: SFT ~4 min, GRPO ~80 min. Total ~$3 per full training run on L40S.

---

## The reward hacking saga — and how the env taught us to fix it

This is the most interesting part.

### Round 1: GRPO looked great

After our first run we reported **SFT 0.62 → SFT+GRPO 0.83 on the hard held-out tier**. Near the gold-pipeline ceiling of 0.92. Popped the champagne.

### Round 2: per-step tracing showed the 0.83 was a hack

We ran each agent with full action traces on both marathons:

| Agent | Marathon 036 | Marathon 037 |
|---|---|---|
| HeuristicAgent | 21 steps, 9 successful Space calls, grade 0.948 | 20 steps, 6 calls, grade 0.918 |
| **SFT+GRPO v1** | **4 steps, 2 calls, grade 0.996** | **4 steps, 2 calls, grade 0.996** |

SFT+GRPO had learned to make 1–2 Space calls, then submit a JSON with every schema key stuffed with the literal string `"synthetic_value_from_previous_step"`. All 12 fields. Identical. 4 steps to 0.996.

The rubric's `expert_score` was stubbed (we planned a real expert reviewer later) and fell back to `format_score` (which only checks "all keys present and non-empty"). Combined with a lenient engagement gate (≥1 successful call), a minimal-work submission scored ~0.99.

### Round 3: three rubric fixes, one search-fix

We added three independent checks:
- **A — Value diversity in format_score**: penalize submissions where >50% of fields have identical canonical values
- **B — Pipeline-aware engagement gate**: require ≥40% of the gold pipeline's Space calls (vs the old ≥1)
- **C — Grounding multiplier**: score fields against observed Space output tokens; poor overlap halves the final score

Also: a parallel bug in the training data — the baseline HeuristicAgent had direct access to `gold_pipeline`, so it never emitted `search_spaces`. SFT learned that shortcut too. We rewrote the HeuristicAgent to always `search → verify → read_card → call`, regenerated 651 training pairs (32.7% are now search actions, was 0%), and retrained SFT for 3 epochs.

### Round 4: retrained under honest scoring

[INSERT FINAL BAR CHART HERE — hardening + search-aware story]

---

## Results

[INSERT two_tier_bar_chart.png — agents vs tiers]

Key numbers (all under the hardened rubric):

| Agent | EASY (5) | HARD (5) | All (10) |
|---|---|---|---|
| Base Qwen 1.5B | 0.15 · 0/5 | 0.15 · 0/5 | 0.15 · 0/10 |
| SFT (search-aware, 3ep) | [v3] | [v3] | [v3] |
| **SFT + GRPO v3** | **[v3]** | **[v3]** | **[v3]** |
| HeuristicAgent (ceiling) | 0.96 · 5/5 | 0.66 · 3/5 | 0.79 · 8/10 |

The heuristic itself dropped from 0.94 → 0.88 under the hardened rubric (one task, `long_meeting_analysis_034`, has a fundamental grounding issue we're tracking — see `REWARD_HACKING_FIX_PLAN.md`).

[INSERT reward_curve_v3.png — training trajectory]

---

## What this demonstrates

1. **Tool-discovery agents need training environments that scale** — 5,002 Spaces is enough to force real search, not memorization
2. **Multi-step RL is required when the env is multi-step** — vanilla TRL GRPOTrainer treats each prompt as one-shot; our custom trajectory-level loop fixed a regression caused by training only on step-0 observations
3. **Reward hacking WILL happen** — and per-task trace inspection is how you catch it. Unit tests won't
4. **An environment's value is the iterate-rubric-alongside-agent loop** — we caught the hack, patched the rubric three ways, retrained, and report honest-lower numbers that actually correspond to solving the task

The env is designed to surface reward-function weaknesses. That's a feature, not a bug.

---

## Future work

- **LLM-judge expert scorer** (Fix D in our plan): real content-quality grading via Claude Haiku or GPT-4o-mini at submit time
- **Shadow-eval during training** (Fix E): detect RL-hacking divergence early via held-out rubric
- **Self-curriculum**: generate harder marathon variants from the agent's own failure modes (Theme #4)

---

## Credits

Built by [Rishabh](https://huggingface.co/rishabh16196) for the Meta PyTorch OpenEnv Hackathon India 2026. Uses Qwen 2.5 1.5B Instruct (Alibaba), Unsloth for training, TRL 0.22.2, HF Jobs L40S.

**All training code, rubric design, trace inspection tools, and training adapters open-source. See [GitHub](https://github.com/rishabh16196/hf-space-composer).**
