"""Why NVARC's selection cannot be fixed by re-weighting: measurements on two seed runs.

  python harness/theory.py runs/s0/cands_seed0.tar runs/s1/cands_seed1.tar

For every test output we take each run's kgmon-ranked candidates and ask:
  1. Confidence: does the kgmon score separate correct top-1 answers from wrong ones?
  2. Reproducibility: when top-1 is wrong, do independent runs produce the SAME wrong grid?
  3. Near-miss: how far are wrong top-1 answers from the truth (shape, % of cells)?
  4. Where: by output-shape relation to the input.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nvarc_decoder as D  # noqa: E402
import score as S  # noqa: E402


def groups(v):
    g = {}
    for c in v.values():
        g.setdefault(D.hashable(c["solution"]), []).append(c)
    return g


def ranked(v):
    """[(grid, kgmon, n_views, mean_aug_nll)] best first."""
    out = [(np.array(h), D.getter_kgmon(gs), len(gs), float(np.mean([np.mean(c["score_aug"]) for c in gs])))
           for h, gs in groups(v).items()]
    return sorted(out, key=lambda x: -x[1])


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main(paths):
    truth = {f"{k}_{i}": np.array(o) for k, outs in S.SOL.items() for i, o in enumerate(outs)}
    test_in = {f"{k}_{i}": np.array(t["input"]) for k, t in S.CH.items() for i, t in enumerate(t["test"])}
    runs = [S.load_cands(p, "") for p in paths]
    R = [{bk: ranked(v) for bk, v in run.items()} for run in runs]
    report = {}

    # 1. confidence separation (per run, top-1 only)
    print("1. Does the selection score know when it is right?  (top-1 candidate per output)")
    for j, rk in enumerate(R):
        pos = [r[0][1] for bk, r in rk.items() if np.array_equal(r[0][0], truth[bk])]
        neg = [r[0][1] for bk, r in rk.items() if not np.array_equal(r[0][0], truth[bk])]
        pv = [r[0][2] for bk, r in rk.items() if np.array_equal(r[0][0], truth[bk])]
        nv = [r[0][2] for bk, r in rk.items() if not np.array_equal(r[0][0], truth[bk])]
        a = auc(pos, neg)
        # how many wrong answers are at least as confident as the median right answer
        frac = float(np.mean(np.array(neg) >= np.median(pos))) if pos and neg else float("nan")
        print(f"   run {j}: right {len(pos):3d} wrong {len(neg):3d}  AUC(kgmon) {a:.3f}  "
              f"wrong >= median-right confidence: {100*frac:.0f}%  "
              f"views: right {np.median(pv):.0f} / wrong {np.median(nv):.0f} (median)")
        report[f"run{j}_auc"] = a
        report[f"run{j}_wrong_ge_median_right"] = frac
        # calibration by kgmon quintile
        allv = sorted([(r[0][1], np.array_equal(r[0][0], truth[bk])) for bk, r in rk.items()], key=lambda x: x[0])
        q = np.array_split(np.array(allv, dtype=float), 5)
        print("          accuracy by confidence quintile (low->high):",
              " ".join(f"{100*c[:, 1].mean():3.0f}%" for c in q))
        report[f"run{j}_quintile_acc"] = [float(c[:, 1].mean()) for c in q]

    # 2. reproducibility of errors across runs
    if len(R) >= 2:
        both = [bk for bk in R[0] if bk in R[1]]
        cats = Counter()
        for bk in both:
            a, b = R[0][bk][0][0], R[1][bk][0][0]
            ra, rb = np.array_equal(a, truth[bk]), np.array_equal(b, truth[bk])
            same = a.shape == b.shape and np.array_equal(a, b)
            if ra and rb:
                cats["both right"] += 1
            elif ra or rb:
                cats["one right"] += 1
            elif same:
                cats["both wrong, SAME grid"] += 1
            else:
                cats["both wrong, different grids"] += 1
        bw = cats["both wrong, SAME grid"] + cats["both wrong, different grids"]
        print("\n2. When top-1 is wrong, do independent runs make the same mistake?")
        for k, v in cats.most_common():
            print(f"   {k:30s} {v:3d}")
        print(f"   -> {100*cats['both wrong, SAME grid']/max(bw,1):.0f}% of shared errors are the identical wrong grid")
        report["error_reproducibility"] = dict(cats)

    # 3. near misses (run 0)
    print("\n3. How wrong are the wrong top-1 answers?  (run 0)")
    shape_wrong, cell_err = 0, []
    for bk, r in R[0].items():
        g, t = r[0][0], truth[bk]
        if np.array_equal(g, t):
            continue
        if g.shape != t.shape:
            shape_wrong += 1
        else:
            cell_err.append(float((g != t).mean()))
    ce = np.array(cell_err)
    print(f"   wrong shape: {shape_wrong}   right shape: {len(ce)}")
    if len(ce):
        print(f"   right-shape errors: median {100*np.median(ce):.1f}% of cells wrong; "
              f"<=2% cells wrong: {int((ce <= .02).sum())}; <=10%: {int((ce <= .10).sum())}")
    report["near_miss"] = {"wrong_shape": shape_wrong, "cell_err": ce.tolist()}

    # 4. by output-shape relation (run 0)
    print("\n4. Where does it fail?  (run 0, by output shape vs test-input shape)")
    tab = {}
    for bk in truth:
        rel = "same shape as input" if truth[bk].shape == test_in[bk].shape else "different shape"
        ok = bk in R[0] and np.array_equal(R[0][bk][0][0], truth[bk])
        orc = bk in runs[0] and any(np.array_equal(c["solution"], truth[bk]) for c in runs[0][bk].values())
        t = tab.setdefault(rel, [0, 0, 0])
        t[0] += 1; t[1] += ok; t[2] += orc
    for rel, (n, ok, orc) in tab.items():
        print(f"   {rel:22s} n={n:3d}  top-1 right {100*ok/n:4.0f}%  answer generated at all {100*orc/n:4.0f}%")
    report["by_shape"] = tab

    out = Path(__file__).parent.parent / "runs" / "theory_report.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
