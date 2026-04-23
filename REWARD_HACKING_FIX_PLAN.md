# Reward Hacking Fix Plan

**Problem**: Trained agents game the rubric by making 1-2 token-efficient Space calls, then submitting with all `expected_output_schema` keys filled with placeholder strings. Grade: 0.99+. Evidence: see `local_training/SLIDE_STORYLINE.md` "Reward Hacking Discovery" section.

**Root cause (ranked)**:

1. `expert_score` in `server/rubrics.py` falls back to `format_score` (binary key-presence) when metadata is empty. It always is — the expert reviewer is a stub.
2. `format_score` only checks "key present, non-empty" — a single literal string fills all fields.
3. Engagement gate requires `successful_space_calls >= 1` — too lenient for marathons that need 6-12 calls.
4. No check that submitted values actually came from observed Space outputs (no provenance/grounding).
5. Value-diversity check missing — a submission with 12 identical strings scores the same as 12 distinct computed outputs.

---

## Fixes, ranked by cost/benefit

Each fix can ship independently; they compound.

### Fix A — Value diversity check in `format_score` *[1 hr, high impact]*

Penalize submissions where most fields have identical or near-identical values. That's the *exact* pattern the trace showed.

```python
# server/rubrics.py::compute_format_score
def compute_format_score(submitted, expected_schema):
    if not expected_schema: return 1.0
    if not isinstance(submitted, dict) or not submitted: return 0.0
    expected_keys = list(expected_schema.keys())
    present = [k for k in expected_keys if k in submitted and submitted[k] not in (None, "", [], {})]
    presence_ratio = len(present) / len(expected_keys)

    # Value-diversity: among present fields, fraction of unique values
    values = [str(submitted[k]).strip().lower() for k in present]
    if len(values) >= 3:
        unique_ratio = len(set(values)) / len(values)
        # If <50% unique, apply diversity penalty
        diversity_mult = unique_ratio if unique_ratio < 0.5 else 1.0
        return presence_ratio * diversity_mult
    return presence_ratio
```

**Expected effect**: marathon SFT/GRPO hack submissions drop from 1.0 → ~0.15 (12 identical values → 1/12 = 0.083 × presence 1.0 ≈ 0.08). Heuristic (with 12 distinct computed outputs) stays at 1.0.

**Risk**: a task with legitimately-similar outputs (e.g., translations that happen to produce near-identical results) would get hit. Mitigation: trigger only when unique_ratio < 0.5 which is very strict.

---

### Fix B — Gold-pipeline-aware engagement gate *[1 hr, complements A]*

Current: gate trips if `successful_space_calls == 0`.
Proposed: gate uses per-task `min_space_calls`, default to `max(2, ceil(len(gold_pipeline) * 0.5))`.

```python
# server/rubrics.py — extend engagement gate
min_calls = meta.get("min_space_calls")
if min_calls is None:
    gold = meta.get("gold_pipeline") or []
    min_calls = max(1, int(len(gold) * 0.5))  # need ≥50% of pipeline

if task_requires_tools and n_successful_calls < min_calls:
    # Cap progressively based on how far short
    shortfall = (min_calls - n_successful_calls) / min_calls
    final_score = min(final_score, 0.15 + 0.35 * (1 - shortfall))
    engagement_gate_applied = True
```

**Expected effect**: marathon gold pipeline has ~9 calls → min=5. SFT's 3 calls → gated. GRPO's 2 calls → gated even harder. Heuristic's 9 calls → unaffected.

**Risk**: hurts legitimate "shortcut" strategies if any exist. Our tasks don't really admit shortcuts (the gold_pipeline is minimal for correctness) so this is safe.

---

### Fix C — Content provenance check *[2 hrs, stronger than A+B]*

Every submitted field must either:
- Contain a substring that appears in a `call_space` output, OR
- Be declared as a composition of prior outputs via a new `sources` payload field

```python
# Track all Space outputs during episode in env
space_output_corpus = set()
for output_record in all_step_outputs:
    space_output_corpus.add(str(output_record["output"]))

# At grading time, for each submitted field
grounded_fields = 0
for k, v in submitted.items():
    if any(tok in corpus_join for tok in extract_key_tokens(v)):
        grounded_fields += 1

grounding_score = grounded_fields / len(submitted)
# New weighted component (could replace or augment format_score)
```

**Expected effect**: placeholder strings like `"synthetic_value_from_previous_step"` won't match any real Space output → grounding=0 → grade capped low.

**Risk**: false-negatives on legitimate cross-domain composition (e.g., "translate this transcript" — the translated output won't match the original transcript word-for-word). Mitigation: use fuzzy token overlap with low threshold (20%).

---

### Fix D — Real LLM-judge expert scorer *[4-6 hrs + API budget]*

The rubric comment already admits this was always the plan. Implement it.

```python
# server/expert_reviewer.py (new or extend existing stub)
def grade_answer(task, submitted, actual_space_outputs):
    prompt = f"""Grade this agent's submission on a 0-1 scale.

    Task: {task.description}
    Expected schema: {task.expected_output_schema}
    Space outputs the agent saw: {actual_space_outputs}
    Submitted answer: {submitted}

    Check:
    - Does each value look like a real answer vs placeholder text?
    - Do values correspond to what the Space outputs suggest?
    - Are translations actually in the right language (not just a flag)?

    Return JSON: {{"score": 0.0-1.0, "issues": [...]}}"""
    response = anthropic.messages.create(
        model="claude-haiku-4-5",  # or similar fast small model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return parse_score(response.content)
```

**Expected effect**: catches placeholder content at the source. Haiku judge at $0.00025/call × 200 training episodes = $0.05 per run — trivial.

**Risk**:
- adds API latency (~500 ms/call) — slows training rollouts
- LLM judge introduces its own biases
- not reproducible across runs without caching

Mitigation: cache judge calls keyed by `(task_id, canonical(submitted))`. Fall back to Fix A+B if the API is unavailable.

---

### Fix E — Shadow-eval during training *[3 hrs, infrastructure]*

Add a held-out **validation tier** that runs every N GRPO steps using the *real* rubric (Fix D), not the cheap training rubric. Surface when the agent's training-rubric score diverges from its validation score — that gap IS the reward-hacking signal, and GRPO can be configured to early-stop when the gap grows.

Less a rubric fix and more a training hygiene pattern. Standard in RLHF.

---

## Recommended rollout

### Today / hackathon-week (1–3 hrs of work)

Ship **Fix A + Fix B** together. They're cheap, share code paths, and together crush the specific hack we observed.

1. Edit `server/rubrics.py` — add value-diversity and gold-pipeline-aware gate
2. Regenerate SFT data against the new rubric (`scripts/generate_gold_trajectories.py`)
3. Re-run `local_training/eval_two_tier.py` on SFT + GRPO adapters we already have
4. Expected numbers: GRPO HARD drops from 0.99 to ~0.5-0.7 (real problem-solving score)
5. **This is more honest to report than the 0.99 headline.**

### Before venue (4-8 hrs)

Add **Fix C (provenance)** — simpler than D, catches most hallucinations without LLM cost.

### At venue (with sponsor compute + API credits)

Add **Fix D (LLM judge)**. Retrain with proper curriculum, report real metrics.

### Stretch

Add **Fix E** shadow-eval for early-stopping on reward-hacking divergence.

---

## Decision matrix

| Fix | Code lines | Time | API $ | Catches | Ship before venue? |
|---|---|---|---|---|---|
| A (value diversity) | ~15 | 1 hr | 0 | filler-text hack ✓ | **YES** |
| B (gold-pipeline gate) | ~15 | 1 hr | 0 | under-called marathons ✓ | **YES** |
| C (provenance) | ~80 | 2 hrs | 0 | hallucinated content | yes if time |
| D (LLM judge) | ~150 | 4-6 hrs | ~$0.05/run | everything, robustly | venue |
| E (shadow-eval) | ~60 | 3 hrs | depends on D | RL-hacking divergence | venue |

---

## Narrative for the pitch deck

> "We designed our env with a minimal engagement gate to block zero-call hacks. After training, per-task tracing revealed a second hack: the agent submits placeholder answers with all schema keys filled. Our rubric's `expert_score` stub fell back to format completeness. We caught this empirically, not through unit tests. Two 15-line fixes patch it before venue compute; a proper LLM-judge is planned for Phase-2."

This is exactly the Theme #2 / #3.1 story: an env that surfaces where the reward is weakest, teaching successive generations of rubrics alongside successive generations of agents.
