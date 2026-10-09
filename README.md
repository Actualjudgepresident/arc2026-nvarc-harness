# ARC Prize 2026 (ARC-AGI-2) — NVARC measurement harness

Offline tooling for measuring where the NVARC solver (ARC Prize 2025 winner) loses points on ARC-AGI-2:
**search coverage** (is the right grid generated at all?) versus **selection** (is it ranked in the top 2?).
The full write-up, with results, is in [WRITEUP.md](WRITEUP.md).

| | seed 0 | seed 1 | pooled |
|---|---|---|---|
| Eval score (top-2) | 28.47% | 26.53% | 29.44% |
| Oracle (right grid among candidates) | 36.25% | 31.67% | 40.42% |

Public leaderboard for seed 0: 27.64.

## Contents

| Path | What |
|---|---|
| `build_fork.py` | Builds our Kaggle notebook from the upstream NVARC notebook via exact, asserted string patches (run all 120 eval tasks, seed offset, raw-candidate export, fallback for empty attempt slots). |
| `harness/score.py` | CPU scorer. `sub`: exact competition metric + union across submissions. `cands`: coverage, oracle and selection accuracy per run and pooled. |
| `harness/dsl.py` | Small CPU program search (depth-2 grid transforms). Negative result: 0/120 eval. |
| `runs/s0`, `runs/s1` | Raw candidates (`cands_seedN.tar`), eval submissions and Kaggle logs of our two runs. |
| `fetch_upstream.sh` | Downloads the inputs we do not redistribute (competition data, upstream notebook, NVARC's `arc_decoder.py`). |

## Reproduce

```bash
pip install -r requirements.txt          # needs ~/.kaggle/kaggle.json, rules accepted
./fetch_upstream.sh
python harness/score.py cands runs/s0/cands_seed0.tar runs/s1/cands_seed1.tar
python harness/score.py sub runs/s0/eval_submission_seed0.json runs/s1/eval_submission_seed1.json
```

New GPU run (Kaggle, 4×L4, internet off, ~5.3 h for the 120 eval tasks):

```bash
python build_fork.py --owner <kaggle-user> --seed 0
kaggle kernels push -p nb/nvarc-eval-s0
```

## Credits

The solver and model are entirely NVARC's (`sorokin/qwen3_4b_grids15_sft139`). The forked notebook is
"ARC AGI2 LB33.89 Minimal Perfpatch" by mikelou1 on Kaggle. Neither is redistributed here;
`fetch_upstream.sh` pulls them. Only the files in this repo are covered by [LICENSE](LICENSE).
