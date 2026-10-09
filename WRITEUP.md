# Measuring Where NVARC Loses Points on ARC-AGI-2: Search Coverage vs. Selection

**Team:** vladee · **Competition:** ARC Prize 2026 – ARC-AGI-2 · **Public LB:** 27.64
**Status:** DRAFT (2026-10-09) – numbers will be updated after the final submission.

## TL;DR

We did not build a new solver. We forked the open-sourced ARC Prize 2025 winner (NVARC) and built an
offline measurement harness around it, so that every idea could be tested on the 120 public evaluation
tasks without spending leaderboard submissions. The harness tracks the leaderboard closely (local 28.47%
vs. public 27.64% for the same configuration). Using it, we decomposed NVARC's error into two parts:

| | seed 0 | seed 1 | seeds 0+1 pooled |
|---|---|---|---|
| **Score (top-2, eval set)** | 28.47% | 26.53% | 29.44% |
| Correct grid anywhere among candidates ("oracle") | 36.25% | 31.67% | 40.42% |
| Test outputs with ≥1 candidate | 162/172 | 154/172 | 164/172 |

Main findings:

1. **Run-to-run variance is large.** Two runs that differ only in random seed differ by ~2 points, and
   solve partly different tasks: the union of their top-2 attempts reaches 31.39%. Public reruns of the
   identical notebook range from 26.9 to 33.9 on the leaderboard. Leaderboard differences of 1–2 points
   between NVARC variants are mostly noise.
2. **NVARC knows *which* outputs it has solved.** Its own selection score separates correct from wrong
   top-1 answers with AUC 0.83 (both seeds). The least-confident fifth of outputs is 0% correct, the most
   confident fifth ~60%.
3. **The ~8 points lost in selection are sampling noise in low-confidence outputs, not a stable bias.**
   In 15 outputs the correct grid was generated but out-ranked. There the winner came from only 1–8
   augmented views and the truth from 1–4. The winning wrong grid recurs as the other seed's top-1 in
   just 2/15 cases. A search over 192 re-weighting formulas gains a net 2 outputs in-sample, i.e. noise:
   no re-weighting of noisy evidence can fix this.
4. **Extra compute helps only where confidence is low.** Pooling a second seed on *all* outputs adds
   1.0 point; pooling it only on the least-confident half adds 2.1 (30.56%), because on confident
   outputs the second run mostly injects competing wrong candidates. (The 50% cut was chosen post hoc on
   172 outputs – a hypothesis, not a validated gain.)
5. **A fixed set of tasks is out of reach.** The same 8 tasks hit the 20-minute per-task cap in both
   seeds (all large ~30×30 outputs with nearly all-distinct candidates, i.e. the model is guessing). Of
   the 10 outputs with zero candidates in seed 0, 8 are also empty in seed 1.

## Base system (not our work)

All credit for the solver goes to the NVARC team (ARC Prize 2025 winner) and to the public Kaggle
notebook "ARC AGI2 LB33.89 Minimal Perfpatch" by mikelou1, which we forked unchanged except for the
instrumentation below. The pipeline:

- Model: `sorokin/qwen3_4b_grids15_sft139`, a Qwen3-4B fine-tuned on ARC-style grids with a reduced
  digit/newline token vocabulary.
- Per task: test-time fine-tuning of a rank-256 LoRA on augmented copies of the task's own train pairs
  (dihedral transforms × color permutations).
- Decoding: depth-first search that keeps every completion with probability ≥ 0.2 (`max_score = -log 0.2`),
  over 16 augmented views of each test input.
- Scoring: each distinct candidate is re-scored by the model's NLL under 8 augmentations.
- Selection: `score_kgmon` = (number of views producing the grid) − (mean augmented NLL); top two
  become attempt_1 / attempt_2.

The perfpatch notebook only changes how logits are gathered on the GPU, which makes it faster; it
does not change any outputs.

## Our contribution: an offline harness

**Fork builder (`build_fork.py`).** Generates our notebook from the upstream one by exact string patches,
each asserted to match exactly once, so an upstream change fails loudly instead of silently producing a
different notebook. Options:

- `ARC_EVAL_ALL` – in a normal commit, run all 120 evaluation tasks (upstream ran only 4). The hidden
  test path used on submission is unchanged.
- `ARC_SEED` – offsets every seed used in TTT, LoRA init and augmentation, for variance studies.
- Exports every raw candidate (grid, beam score, 8 augmented scores) as `cands_seedN.tar`.
- `ARC_FALLBACK` / `ARC_FALLBACK_MIN` – see "Untested idea" below.

**Scorer (`harness/score.py`).** Runs on a laptop CPU, no GPU needed.
- `sub` mode: the exact competition metric (task-averaged, 2 attempts), plus a format check and the
  union across several submissions.
- `cands` mode: loads raw candidates from one or more runs. For each run and for the pooled set it
  reports coverage, oracle accuracy, and the top-2 accuracy of each selection algorithm. It reuses
  NVARC's own `arc_decoder.py` verbatim, so the selection numbers match the notebook's.

This turns every selection or pooling idea into a free CPU experiment. Only changes to search or
training need GPU time.

**A note on the notebook's printed score.** Upstream prints `acc: 34.2/116`, but that divides by the 116
tasks that received any candidate, not by all 120. The true eval score of that run is 28.47%.

**Contamination check.** If the base model had been trained on the evaluation tasks, local scores would
sit well above the leaderboard. They don't (28.47 local vs. 27.64 public), so we treat the evaluation
set as a valid proxy.

## Why it behaves this way (theory)

All numbers below come from `harness/theory.py` on the two seed runs (`runs/theory_report.json`).

**1. The confidence signal is real.** `score_kgmon` (views producing the grid minus mean augmented NLL)
is a well-calibrated confidence: AUC 0.83 for "top-1 is correct", accuracy rising from 0% to ~60% across
confidence quintiles. Correct top-1 answers are produced by a median of 8–9 views, wrong ones by 2–3.
When the model has understood a rule, the right answer dominates every view.

**2. Errors come in two kinds.** Where both seeds' top-1 answers are wrong (109 outputs), 33% are the
*identical* wrong grid – a reproducible misreading of the rule that more sampling cannot fix – and 67%
differ, i.e. guessing. Of the right-shape wrong answers, 59/107 differ from the truth in at most 10% of
cells (17 in at most 2%): many failures are near-misses on a mostly understood rule.

**3. Lost-in-selection = low evidence.** The outputs where the truth is generated but out-ranked are
exactly the low-confidence ones: a handful of views on each side, and the winner does not reproduce
across seeds (2/15). The ranking is not biased; the evidence is just too thin to rank on. That is why
192 re-weightings fail, and why pooling a second run helps only where evidence was thin.

**Implications.** (a) Compute should be allocated by confidence: re-run or widen the search only on
low-confidence outputs, since confident outputs are mostly right already and only lose from extra noise.
(b) The reproducible-misreading errors (a third of shared errors) need evidence the model does not
produce itself – e.g. an independent verifier such as executable programs checked against the train
pairs, or a stronger model. (c) Near-misses suggest local repair (editing a few cells of a high-confidence
candidate) as a cheap target.

## Things that did not work

- **Re-weighting selection** (192 formulas): +0.84 points at best, in-sample, i.e. noise.
- **Two seeds pooled:** +1.0 point on the eval set, but two full runs do not fit the 12-hour limit on the
  240 hidden tasks (one run ≈ 21 GPU-hours on 4 GPUs ≈ 5.3 h wall-clock for 120 tasks ≈ 10.7 h for 240).
- **Cheap CPU program search** (`harness/dsl.py`): depth-2 compositions of whole-grid transforms plus a
  learned color map. It solves 43/1000 training tasks but **0/120** evaluation tasks. Simple global
  transforms do not cover ARC-AGI-2.

## Untested idea: filling unused attempt slots

In seed 0, 10 outputs had no candidate and 11 had only one, so 31 attempt slots were submitted as
`[[0]]`. 19 outputs had fewer than two candidates in *both* seeds. `ARC_FALLBACK_MIN=2` adds greedy
decodes of 4 augmented views for exactly those outputs, scored like any other candidate. It can only fill
empty slots, so it cannot lower the score. A probe on these 15 tasks plus 3 controls is built but has not
run yet (GPU quota). Expected effect: 0–3 outputs.

## Rubric self-assessment

- **Accuracy:** ~28 public. Below the NVARC pack's best draws. No claim here.
- **Universality:** the harness – raw-candidate export, oracle-vs-selection decomposition, multi-run
  pooling – applies to any sample-then-select solver on any exact-match benchmark.
- **Progress:** the main value is negative results with numbers attached. Selection tuning and seed
  pooling are near their ceiling for this model; gains must come from a stronger generator or an
  independent verifier.
- **Theory:** measured, not asserted: calibration (AUC 0.83), error reproducibility across seeds (33%
  identical), near-miss structure, and the low-evidence nature of selection losses.
- **Completeness:** all code, both runs' raw candidates and logs will be released (see below).
- **Novelty:** low. The method is NVARC; the contribution is measurement.

## Reproducibility

- Code: `build_fork.py`, `harness/score.py`, `harness/nvarc_decoder.py`, `harness/dsl.py` –
  https://github.com/Actualjudgepresident/arc2026-nvarc-harness
- Notebooks: `vladee/nvarc-eval-s0`, `vladee/nvarc-eval-s1` – **[TODO: make public]**
- Raw candidates: `cands_seed0.tar`, `cands_seed1.tar` (~4 MB each).
- Rebuild: `python build_fork.py --owner <you> --seed 0`, push to Kaggle (4×L4, internet off), then
  `python harness/score.py cands cands_seed0.tar`.

## Acknowledgements and licenses

NVARC solver and model: NVARC team / sorokin (model listed as Apache-2.0 – **[TODO: verify both licenses
on the model page and the NVARC repo]**). Upstream notebook: mikelou1 ("ARC AGI2 LB33.89
Minimal Perfpatch"). Our additions: MIT (see LICENSE).
