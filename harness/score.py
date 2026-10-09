"""Local ARC-AGI-2 eval harness (120 eval tasks / 172 outputs).

  Score submissions (exact competition metric; union = any attempt of any file):
    python harness/score.py sub sub_a.json [sub_b.json ...]

  Score raw NVARC candidate dirs/tars (the cands_seed*.tar our fork exports):
    python harness/score.py cands cands_seed0.tar [cands_seed1.tar ...]
  reports per run and pooled: oracle coverage (correct grid anywhere among
  candidates), and top-2 accuracy of each selection algorithm.
"""
import argparse
import bz2
import json
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nvarc_decoder as D  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
CH = json.load(open(DATA / "arc-agi_evaluation_challenges.json"))
SOL = json.load(open(DATA / "arc-agi_evaluation_solutions.json"))
N_TASKS = len(SOL)


def weight(key):  # task-averaged, as in the metric
    return 1 / len(SOL[key])


def solved_outputs(sub):
    """{task_i}: outputs where attempt_1 or attempt_2 is exactly right."""
    got = set()
    for k, outs in SOL.items():
        preds = sub.get(k, [])
        for i, truth in enumerate(outs):
            if i < len(preds) and any(preds[i].get(a) == truth for a in ("attempt_1", "attempt_2")):
                got.add(f"{k}_{i}")
    return got


def check_format(sub):
    missing = set(CH) - set(sub)
    bad = [k for k in CH if k in sub and (len(sub[k]) != len(CH[k]["test"]) or
           any(set(p) != {"attempt_1", "attempt_2"} for p in sub[k]))]
    return missing, bad


def pct(outputs):
    return 100 * sum(weight(o.rsplit("_", 1)[0]) for o in outputs) / N_TASKS


def cmd_sub(files):
    union = set()
    for f in files:
        sub = json.load(open(f))
        missing, bad = check_format(sub)
        s = solved_outputs(sub)
        union |= s
        flag = f"  FORMAT: {len(missing)} missing, {len(bad)} malformed" if missing or bad else ""
        print(f"{pct(s):6.2f}%  ({len(s)}/172 outputs)  {f}{flag}")
    if len(files) > 1:
        print(f"{pct(union):6.2f}%  union (upper bound if attempts were chosen perfectly)")


def load_cands(path, run_name):
    """Load one run's bz2 pickles (dir or tar) into a decoder-style dict."""
    res = {}
    if not Path(path).is_dir():
        tar = tarfile.open(path)
        items = [(Path(m.name).name, tar.extractfile(m)) for m in tar.getmembers() if m.isfile()]
    else:
        items = [(p.name, open(p, "rb")) for p in Path(path).iterdir() if p.is_file()]
    for name, fh in items:
        outputs = pickle.loads(bz2.decompress(fh.read()))
        bk = name.split(".")[0]
        for i, sample in enumerate(outputs):
            res.setdefault(bk, {})[f"{name}{run_name}.out{i}"] = sample
    return res


def eval_cands(res, label):
    truth = {f"{k}_{i}": np.array(o) for k, outs in SOL.items() for i, o in enumerate(outs)}
    oracle = {bk for bk, v in res.items() if any(np.array_equal(g["solution"], truth[bk]) for g in v.values())}
    line = [f"{label:28s} covered {len(res):3d}/172  oracle {pct(oracle):6.2f}%"]
    for algo in D.selection_algorithms:
        sel = {bk: algo(v) for bk, v in res.items()}
        ok = {bk for bk, g in sel.items() if any(np.array_equal(x, truth[bk]) for x in g[:2])}
        line.append(f"{algo.__name__.replace('score_', '')} {pct(ok):6.2f}%")
    print("  ".join(line))


def cmd_cands(paths):
    pooled = {}
    for j, p in enumerate(paths):
        res = load_cands(p, run_name=f".run{j}")
        eval_cands(res, Path(p).name)
        for bk, v in res.items():
            pooled.setdefault(bk, {}).update(v)
    if len(paths) > 1:
        eval_cands(pooled, f"POOLED x{len(paths)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sub", "cands"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    (cmd_sub if a.mode == "sub" else cmd_cands)(a.paths)
