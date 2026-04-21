# Spaces Pipeline Pro — 3-Minute Pitch Outline

## Slide 1 (0:00-0:20) — Hook
**Title:** "We taught an AI to use HuggingFace fluently."

**Visual:** HuggingFace logo with 500K Spaces stat, then GAIA leaderboard showing frontier models stuck at 33-44%.

**Script:**
> "HuggingFace Hub is the world's biggest catalog of AI tools — 500,000 Spaces. But frontier models like GPT-5 and Claude 4.5 score only 33% on benchmarks that require composing multiple tools. The bottleneck isn't model size. It's that nobody has built a training environment for this."

---

## Slide 2 (0:20-0:50) — Baseline Failure
**Title:** "What untrained agents do today"

**Visual:** Live screen recording of untrained Qwen 1.5B given a task: agent picks wrong Space, ignores Auditor flag, submits empty answer. Score: **0.18**.

**Script:**
> "Watch an untrained Qwen 1.5B try to transcribe Hindi audio and summarize. It calls a decoy Space. The Auditor flags it. The agent ignores the flag, submits garbage. Final score: 0.18. This is what tool-use looks like before training."

---

## Slide 3 (0:50-1:20) — Training Curves
**Title:** "What 30K rollouts of GRPO does"

**Visual:** 4 reward curves (one per training phase) rising across 30K steps. Show:
- Final reward: 0.2 → 0.7
- Auditor flags/episode: 8 → 1.5
- Avg actions to submit: 18 → 7
- Schema drift recovery rate: 20% → 75%

**Script:**
> "We trained the same model with TRL GRPO + Unsloth on our environment. Four phases: warmup, discovery, drift, adversarial. By phase 4, the agent uses 60% fewer actions, triggers 80% fewer Auditor flags, and recovers from API contract changes 75% of the time."

---

## Slide 4 (1:20-2:30) — Live Demo
**Title:** "Watch it use real HuggingFace Spaces"

**Visual:** Three-pane view:
- Left: agent's reasoning + JSON actions
- Right: live Space outputs streaming in (real audio playing, transcript appearing, translations rendering)
- Bottom: Auditor flag log

**Demo flow** (~70 seconds, choose ONE task live):
1. Task: "Spanish audio → English transcript → sentiment → French summary"
2. Agent searches for transcription Space (3 sec)
3. Reads whisper card (1 sec)
4. Calls Whisper on real audio (8 sec) — transcript appears
5. Calls NLLB translation (5 sec) — English text appears
6. Calls sentiment classifier (2 sec) — "neutral"
7. Calls BART summarizer (4 sec) — summary appears
8. Calls NLLB to French (5 sec)
9. **Mid-demo simulate drift event**: announce "NLLB has been deprecated, use NLLB-v2"
10. Agent re-reads card, switches to successor, succeeds
11. Submits, score: **0.78**

**Script:**
> "Here's the trained agent on a fresh task. It discovers Whisper from the catalog... reads its card... calls it on real audio... [live audio playing] ... transcript comes back ... now translation ... sentiment ... summary ... [DRIFT EVENT] — and look: we just deprecated NLLB mid-demo. The agent gets an error, re-reads the card, finds the successor, and recovers. Final score: 0.78."

---

## Slide 5 (2:30-2:50) — Sub-Theme Coverage
**Title:** "Stacks 4 sub-themes in one environment"

**Visual:** 4 colored blocks with sponsor logos:
- 🟢 **Halluminate** (Multi-Actor)
- 🔵 **Fleet AI** (Oversight Auditor)
- 🟠 **Patronus** (Schema Drift)
- 🟣 **Snorkel** (Evolving Expert)

**Script:**
> "This environment hits four sub-theme bonuses in one design: Halluminate for multi-actor orchestration, Fleet AI for the oversight Auditor, Patronus for the schema drift mechanic, and Snorkel for the evolving expert reviewer with shifting personas."

---

## Slide 6 (2:50-3:00) — Close
**Title:** "Open source. Built on OpenEnv."

**Visual:** GitHub URL + HF Space URL.

**Script:**
> "Full code, 25 tasks, training scripts, and the trained agent are open source. github.com/rishabh16196/spaces-pipeline-env. Thank you."

---

## Q&A Cheat Sheet (2 minutes Q&A)

**Q: How is this different from smolagents?**
A: smolagents is a *framework* for building agents that call Spaces. We built the *training environment* with reward signal, multi-actor oversight, and adversarial mechanics. Complementary — smolagents could be the inference framework on top of our env.

**Q: How is it different from GAIA?**
A: GAIA is a static eval set of 466 questions. We're an interactive RL training env with reward shaping, curriculum, and trainable policies. Solves the training problem GAIA exposes.

**Q: How do you handle live Space failures during the demo?**
A: Three layers of fallback: (1) pre-recorded demo video, (2) mock-mode demo (deterministic, network-free), (3) 3 alternative tasks ready to swap.

**Q: Multi-agent learning, or just one?**
A: One learning agent + multiple non-learning actors (Spaces, Auditor, Expert). v2 design accommodates 2 learning agents (Planner + Executor) without rewrite.

**Q: Why Qwen 1.5B and not bigger?**
A: Trains in free Colab T4. Showed cleaner reward curves with smaller model (more headroom to learn). Can swap to 7B at venue with sponsor compute.

**Q: How do you prevent reward hacking?**
A: Auditor catches obvious gaming (decoy Spaces, redundant calls, empty submissions). Expert with persona shifts prevents overfitting to one reward shape. Held-out tasks measure generalization.

**Q: Is the schema drift realistic?**
A: Yes — based on real HF library breaking changes (e.g., `huggingface_hub` 0.x → 0.20 renamed multiple `ModelInfo` fields). We have 5 drift types: rename, type change, new required, output change, deprecation.
