"""Fit the empirical prior over pair accuracy for src_eb/model.py.

Training pairs with >= 80 distinct items, each benchmark weighted equally; each pair contributes its
Beta(k+1, n-k+1) posterior (moment-matched normal on the probability scale, floored at BANDWIDTH)
on a 100-point grid. Writes <out_dir>/eb_prior.json.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BANDWIDTH = 0.03
GRID = (np.arange(100) + 0.5) / 100


def fit(exclude=frozenset(), out_dir=None, cache_dir=None):
    here = Path(__file__).resolve().parent
    cache_dir = Path(cache_dir or here.parents[1] / "data" / "cache")
    out_dir = Path(out_dir or here / "state")
    out_dir.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(cache_dir / "pairs.parquet")
    P = P[~P.benchmark_id.isin(set(exclude)) & (P["items"] >= 80)]
    a = ((P.k + 1) / (P.n + 2)).to_numpy()
    sd = np.sqrt(a * (1 - a) / (P.n.to_numpy() + 3))
    sd = np.sqrt(sd ** 2 + BANDWIDTH ** 2)
    w = (1.0 / P.groupby("benchmark_id").benchmark_id.transform("size")).to_numpy()
    dens = np.zeros_like(GRID)
    for ai, si, wi in zip(a, sd, w):
        d = np.exp(-0.5 * ((GRID - ai) / si) ** 2)
        dens += wi * d / d.sum()
    dens /= dens.sum()
    state = {"grid": GRID.round(4).tolist(), "prior": dens.round(8).tolist(),
             "n_pairs": int(len(P)), "n_benchmarks": int(P.benchmark_id.nunique()),
             "prior_mean": float((dens * GRID).sum())}
    (out_dir / "eb_prior.json").write_text(json.dumps(state))
    return state


if __name__ == "__main__":
    s = fit()
    print({k: v for k, v in s.items() if k not in ("grid", "prior")})
