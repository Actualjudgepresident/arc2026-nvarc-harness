#!/usr/bin/env bash
# Pulls third-party inputs that this repo does not redistribute:
#  1. competition data (needs a Kaggle account that accepted the ARC Prize 2026 rules)
#  2. the upstream NVARC notebook we fork, and NVARC's arc_decoder.py extracted from it
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data kernels
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p data
(cd data && unzip -oq arc-prize-2026-arc-agi-2.zip && rm arc-prize-2026-arc-agi-2.zip)
kaggle kernels pull mikelou1/arc-agi2-lb33-89-minimal-perfpatch -p kernels/arc-agi2-lb33-89-minimal-perfpatch -m
python3 - <<'PY'
import json, glob
nb = json.load(open(glob.glob("kernels/arc-agi2-lb33-89-minimal-perfpatch/*.ipynb")[0]))
for c in nb["cells"]:
    s = "".join(c["source"])
    if s.startswith("%%writefile arc_decoder.py"):
        open("harness/nvarc_decoder.py", "w").write("# Extracted verbatim from the upstream notebook (arc_decoder.py). Do not edit.\n" + s.split("\n", 1)[1])
        print("wrote harness/nvarc_decoder.py")
PY
