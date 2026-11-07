"""Grid over PAIEC_CONFIG overrides; each point is one eval/simulate.py run. Appends to eval/results/tune_<name>.csv.

    python eval/tune.py submissions/v1 grid.json name
"""
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model, grid_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
grid = json.loads(Path(grid_path).read_text())
keys = sorted(grid)
out = ROOT / "eval" / "results" / f"tune_{name}.csv"
for vals in itertools.product(*(grid[k] for k in keys)):
    cfg = dict(zip(keys, vals))
    tag = "tune_" + name + "_" + "_".join(f"{k}{v}" for k, v in cfg.items())
    env = dict(os.environ, PAIEC_CONFIG=json.dumps(cfg))
    subprocess.run([sys.executable, str(ROOT / "eval" / "simulate.py"), model, "--tag", tag, *extra],
                   env=env, check=True, capture_output=True)
    split = "benchmark" if "benchmark" in extra else "pair"
    res = json.loads((ROOT / "eval" / "results" / f"{Path(model).name}_{split}_k7_s0_{tag}.json").read_text())
    line = json.dumps(cfg) + "\t" + "\t".join(f"{res[k]:.4f}" for k in ["ALC", "B0", "B1", "B3", "B7", "B15", "B31"])
    with open(out, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
