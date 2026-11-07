"""Paired ALC difference between two runs on the same pairs: mean, standard error, and win rate.

    python eval/paired.py <base_pairs.csv> <new_pairs.csv> [...more new]
"""
import sys

import numpy as np
import pandas as pd

W = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
B = [0, 1, 3, 7, 15, 31]


def alc(path):
    d = pd.read_csv(path)
    d["ALC"] = sum(w * d[f"B{b}"] for w, b in zip(W, B))
    return d.set_index(["fold", "benchmark", "subject"]).ALC


base = alc(sys.argv[1])
for f in sys.argv[2:]:
    new = alc(f).reindex(base.index)
    d = (new - base).dropna()
    bench = d.groupby(level="benchmark").mean()
    print(f"{f.split('/')[-1][:60]:60s} diff {d.mean():+.5f} (se {d.std(ddof=1) / np.sqrt(len(d)):.5f}, "
          f"bench-clustered se {bench.std(ddof=1) / np.sqrt(len(bench)):.5f}) better in {np.mean(d < 0):.0%} of pairs")
