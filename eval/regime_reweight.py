"""What-if analysis: ALC of saved runs when the held-out pairs are reweighted so that their mean
evaluation-pool accuracy equals a target (e.g. 0.35, the hidden-test level other teams report).
Weights w = exp(-lam * base_rate), lam solved by bisection. Prints unweighted and reweighted ALC.

    python eval/regime_reweight.py 0.35 <pairs.csv> [<pairs.csv> ...]
"""
import sys

import numpy as np
import pandas as pd

W = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
B = [0, 1, 3, 7, 15, 31]


def weights(br, target):
    lo, hi = -50.0, 50.0
    for _ in range(100):
        lam = (lo + hi) / 2
        w = np.exp(-lam * br)
        if (w * br).sum() / w.sum() > target:
            lo = lam
        else:
            hi = lam
    return w / w.sum()


target = float(sys.argv[1])
for f in sys.argv[2:]:
    d = pd.read_csv(f)
    alc = sum(wt * d[f"B{b}"] for wt, b in zip(W, B))
    w = weights(d.base_rate.to_numpy(), target)
    ess = 1.0 / (w ** 2).sum()
    print(f"{f.split('/')[-1][:70]:70s} ALC {alc.mean():.4f}  reweighted(mean acc {target}) {np.sum(w * alc):.4f}  "
          f"B0 {np.sum(w * d.B0):.4f}  B1 {np.sum(w * d.B1):.4f}  ESS {ess:.0f}")
