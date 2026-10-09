"""Build our NVARC fork from the public LB33.89 notebook.

Every change is an exact string replacement, asserted to hit exactly once, so an
upstream edit fails loudly instead of silently producing a different notebook.

Usage: python build_fork.py [--seed N] [--slug nvarc-eval]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "kernels/arc-agi2-lb33-89-minimal-perfpatch"

CONFIG_CELL = '''# === Fork config ===
# ARC_EVAL_ALL=1: a normal commit solves all 120 evaluation tasks (not just 4) and
#   saves every raw candidate, so selection/pooling can be studied offline.
#   The hidden-test path taken on submission (KAGGLE_IS_COMPETITION_RERUN) is unchanged.
# ARC_SEED: offset added to the TTT / LoRA / augmentation seeds (0 = stock NVARC).
import os
os.environ["ARC_EVAL_ALL"] = "{eval_all}"
os.environ["ARC_SEED"] = "{seed}"
# ARC_FALLBACK=1: if the 20%-probability DFS finds NO candidate for a test output, add greedy decodes
#   of 4 augmentations, scored like any other candidate (only touches outputs that would score 0).
# ARC_TASKS: comma-separated eval task ids to run in commit mode (empty = all). Ignored on rerun.
os.environ["ARC_FALLBACK"] = "{fallback}"
os.environ["ARC_TASKS"] = "{tasks}"
# ARC_FALLBACK_MIN: run the fallback when an output has fewer than this many distinct candidates (1 = empty only).
os.environ["ARC_FALLBACK_MIN"] = "{fallback_min}"
'''

EXPORT_CELL = '''# === Fork: export raw candidates for offline analysis ===
import os, tarfile
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    seed = os.environ["ARC_SEED"]
    with tarfile.open(f"/kaggle/working/cands_seed{seed}.tar", "w") as tar:
        tar.add("/kaggle/inference_outputs", arcname=f"cands_seed{seed}")
    os.system(f"cp submission.json /kaggle/working/eval_submission_seed{seed}.json")
    print("exported", len(os.listdir("/kaggle/inference_outputs")), "candidate files")
'''

SEED = 'int(os.getenv("ARC_SEED", "0"))'

REPLACEMENTS = [
    # all 120 eval tasks in commit mode
    ('        if not rerun_mode:\n            if key not in',
     '        if not rerun_mode and os.getenv("ARC_EVAL_ALL") != "1":\n            if key not in'),
    # seed offsets
    ('        random_state=42,', f'        random_state=42 + {SEED},'),
    ('        seed=42,\n        report_to="none",', f'        seed=42 + {SEED},\n        report_to="none",'),
    ('puzzle_ds.augment(n=16, shfl_keys=True, seed=1)', f'puzzle_ds.augment(n=16, shfl_keys=True, seed=1 + {SEED})'),
    ('puzzle_ds_multi.augment(n=2, seed=2)', f'puzzle_ds_multi.augment(n=2, seed=2 + {SEED})'),
]
EXTRA = [('def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):', 'def greedy_result(model, prefix_tokens, max_new_tokens):\n    """Fork fallback: plain greedy decode per prompt, in the same (batch_id, [(score, tokens)]) shape as DFS."""\n    result = []\n    for i, p in enumerate(prefix_tokens):\n        ids = torch.tensor([p], device=model.device, dtype=torch.long)\n        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,\n                             do_sample=False, eos_token_id=EOS_ID, pad_token_id=PAD_ID,\n                             output_scores=True, return_dict_in_generate=True)\n        gen = out.sequences[0, ids.size(1):].tolist()\n        nll = sum(-torch.log_softmax(sc[0].float(), -1)[t].item() for t, sc in zip(gen, out.scores))\n        result.append((i, [(nll, gen)]))\n    return result\n\n\ndef inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):'), ('            for subkeys in batches:\n\n                spend_time = time.time() - start_time', '            pending = [(b, max_score) for b in batches]\n            fallback_done = set()\n            while pending:\n                subkeys, cur_max_score = pending.pop(0)\n\n                spend_time = time.time() - start_time'), ('                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)', '                if cur_max_score is None:\n                    dfs_result = greedy_result(model, tokens, max_new_tokens)\n                else:\n                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, cur_max_score, end_time)'), ('                            pickle.dump(decoded_result, f)\n\n        memory_allocated', '                            pickle.dump(decoded_result, f)\n\n                if not pending and os.getenv("ARC_FALLBACK") == "1":\n                    saved = os.listdir(dir_outputs)\n                    for tid, sks in test_id_to_subkeys.items():\n                        if tid in fallback_done:\n                            continue\n                        distinct = set()\n                        for fn in saved:\n                            if fn.startswith(f"{key}_{tid}."):\n                                with bz2.BZ2File(os.path.join(dir_outputs, fn)) as fh:\n                                    for smp in pickle.load(fh):\n                                        distinct.add(smp["solution"].tobytes() + bytes(smp["solution"].shape))\n                        if len(distinct) >= int(os.getenv("ARC_FALLBACK_MIN", "1")):\n                            continue\n                        fallback_done.add(tid)\n                        print(f"[Rank {rank}] fallback greedy for {key}_{tid}")\n                        pending.append(([sks[j] for j in (0, 4, 8, 12) if j < len(sks)], None))\n\n        memory_allocated'), ('        queue.put(key)\n    for _ in range(4):', '        if not rerun_mode and os.getenv("ARC_TASKS") and key not in os.getenv("ARC_TASKS").split(","):\n            continue\n        queue.put(key)\n    for _ in range(4):')]



def build(seed, slug, eval_all, owner, fallback=False, tasks="", fallback_min=1):
    src_nb = json.load(open(next(SRC.glob("*.ipynb"))))
    nb = copy.deepcopy(src_nb)
    cells = nb["cells"]
    for old, new in REPLACEMENTS + EXTRA:
        hits = [c for c in cells if c["cell_type"] == "code" and old in "".join(c["source"])]
        assert len(hits) == 1 and "".join(hits[0]["source"]).count(old) == 1, f"patch did not match once: {old!r}"
        hits[0]["source"] = "".join(hits[0]["source"]).replace(old, new)

    def code_cell(s):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    cells.insert(first_code, code_cell(CONFIG_CELL.format(eval_all=int(eval_all), seed=seed, fallback=int(fallback), tasks=tasks, fallback_min=fallback_min)))
    cells.append(code_cell(EXPORT_CELL))
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    out = ROOT / "nb" / slug
    out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / f"{slug}.ipynb", "w"), indent=1)

    meta = json.load(open(SRC / "kernel-metadata.json"))
    for k in ("id_no",):
        meta.pop(k, None)
    meta.update(id=f"{owner}/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True)
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--no-eval-all", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="greedy fallback for outputs with no DFS candidate")
    ap.add_argument("--tasks", default="", help="comma-separated eval task ids (commit mode only)")
    ap.add_argument("--fallback-min", type=int, default=1, help="fallback when an output has < N distinct candidates")
    a = ap.parse_args()
    build(a.seed, a.slug or f"nvarc-eval-s{a.seed}", not a.no_eval_all, a.owner, a.fallback, a.tasks, a.fallback_min)
