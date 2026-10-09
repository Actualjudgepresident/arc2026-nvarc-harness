"""Tiny CPU program search for NVARC's unused attempt slots.

Searches compositions (depth <= 2) of whole-grid transforms, keeps any program that
reproduces EVERY train pair exactly, and applies it to the test inputs. Output only
ever fills slots NVARC left empty, so it cannot lower the score.

  python harness/dsl.py            # report solves on eval + train sets
"""
import itertools
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "data"


def _bg(g):
    v, c = np.unique(g, return_counts=True)
    return v[c.argmax()]


def crop_nonbg(g):
    ys, xs = np.nonzero(g != _bg(g))
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else g


def largest_obj_bbox(g):
    from scipy.ndimage import label
    bg = _bg(g)
    best = None
    for c in np.unique(g):
        if c == bg:
            continue
        lab, n = label(g == c)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if best is None or len(ys) > best[0]:
                best = (len(ys), ys.min(), ys.max(), xs.min(), xs.max())
    if best is None:
        return g
    _, y0, y1, x0, x1 = best
    return g[y0:y1 + 1, x0:x1 + 1]


def halves(g):
    h, w = g.shape
    out = {}
    if h % 2 == 0:
        out["top"], out["bottom"] = g[:h // 2], g[h // 2:]
    if w % 2 == 0:
        out["left"], out["right"] = g[:, :w // 2], g[:, w // 2:]
    return out


def make_unary():
    T = {
        "id": lambda g: g,
        "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 3),
        "flipud": np.flipud, "fliplr": np.fliplr, "transpose": lambda g: g.T, "antitranspose": lambda g: np.rot90(g, 2).T,
        "crop": crop_nonbg, "largest_obj": largest_obj_bbox,
        "mirror_h": lambda g: np.hstack([g, np.fliplr(g)]), "mirror_v": lambda g: np.vstack([g, np.flipud(g)]),
        "mirror_4": lambda g: np.vstack([np.hstack([g, np.fliplr(g)]), np.flipud(np.hstack([g, np.fliplr(g)]))]),
        "sym_lr": lambda g: np.where(g == _bg(g), np.fliplr(g), g), "sym_ud": lambda g: np.where(g == _bg(g), np.flipud(g), g),
        "dedup_rows": lambda g: g[[0] + [i for i in range(1, len(g)) if not np.array_equal(g[i], g[i - 1])]],
        "dedup_cols": lambda g: g[:, [0] + [j for j in range(1, g.shape[1]) if not np.array_equal(g[:, j], g[:, j - 1])]],
    }
    for k in (2, 3, 4):
        T[f"up{k}"] = lambda g, k=k: np.kron(g, np.ones((k, k), dtype=g.dtype))
        T[f"tile{k}"] = lambda g, k=k: np.tile(g, (k, k))
        T[f"down{k}"] = lambda g, k=k: g[::k, ::k]
    for name in ("top", "bottom", "left", "right"):
        T[name] = lambda g, name=name: halves(g)[name]
    for op in ("and", "or", "xor"):
        def f(g, op=op, axis=0):
            hv = halves(g)
            for a, b in (("top", "bottom"), ("left", "right")):
                if a in hv:
                    x, y = hv[a] != _bg(g), hv[b] != _bg(g)
                    m = {"and": x & y, "or": x | y, "xor": x ^ y}[op]
                    return m.astype(g.dtype)
            raise ValueError
        T[f"half_{op}"] = f
    return T


UNARY = make_unary()


def fit_colormap(pairs):
    """Learn a consistent cell-wise color map (incl. identity) over all pairs, or None."""
    m = {}
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        for x, y in zip(a.ravel(), b.ravel()):
            if m.setdefault(int(x), int(y)) != y:
                return None
    return m


def apply_colormap(g, m):
    out = g.copy()
    for x, y in m.items():
        out[g == x] = y
    return out


def run(prog, g):
    for name in prog:
        g = UNARY[name](g)
        if g.size == 0 or g.shape[0] > 30 or g.shape[1] > 30:
            raise ValueError
    return g


def solve(task, max_programs=2):
    """Return up to `max_programs` distinct predictions per test input."""
    train = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    tests = [np.array(p["input"]) for p in task["test"]]
    names = list(UNARY)
    progs = [(n,) for n in names] + [(a, b) for a, b in itertools.product(names, names) if a != "id" and b != "id"]
    found = []
    for prog in progs:
        try:
            outs = [run(prog, a) for a, _ in train]
        except Exception:
            continue
        cmap = None
        if not all(o.shape == b.shape and np.array_equal(o, b) for o, (_, b) in zip(outs, train)):
            cmap = fit_colormap([(o, b) for o, (_, b) in zip(outs, train)])
            if cmap is None or all(x == y for x, y in cmap.items()):
                continue
        try:
            preds = [run(prog, t) for t in tests]
            if cmap is not None:
                if any(int(v) not in cmap for p in preds for v in np.unique(p)):
                    continue
                preds = [apply_colormap(p, cmap) for p in preds]
        except Exception:
            continue
        preds = [p.tolist() for p in preds]
        if preds not in [f[1] for f in found]:
            found.append((prog, preds))
        if len(found) >= max_programs:
            break
    return found


if __name__ == "__main__":
    for split in ("evaluation", "training"):
        ch = json.load(open(DATA / f"arc-agi_{split}_challenges.json"))
        sol = json.load(open(DATA / f"arc-agi_{split}_solutions.json"))
        tasks_hit, outputs_hit, wrong = [], 0, 0
        for k, t in ch.items():
            found = solve(t)
            if not found:
                continue
            ok = [any(f[1][i] == s for f in found) for i, s in enumerate(sol[k])]
            if any(ok):
                tasks_hit.append((k, found[0][0]))
            wrong += not all(ok)
            outputs_hit += sum(ok)
        print(f"{split}: {len(tasks_hit)} tasks with a correct output ({outputs_hit} outputs), "
              f"{wrong} tasks where a fitted program was wrong")
        if split == "evaluation":
            print("  ", tasks_hit)
