"""Leave-benchmarks-out probe: can hashed item features predict item difficulty on unseen benchmarks?

For each pool item: target = smoothed logit of mean response over the pool subjects. Ridge on
hashed features with equal weight per benchmark; same 7 folds as eval/simulate.py. Reports the
correlation with the truth overall, between benchmark means, and within benchmarks.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "eval"))
from item_features import DIM, featurize  # noqa: E402
from simulate import fold_assignment  # noqa: E402


def main():
    rows, cols, vals, tgt, bid, wts = [], [], [], [], [], []
    r = 0
    for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
        pool = pickle.load(open(p, "rb"))
        b = pool["benchmark_id"]
        stats = {}
        for pr in pool["pairs"]:
            for it, inter, y in pr["resp"]:
                s = stats.setdefault(it, [0, 0, inter])
                s[0] += y
                s[1] += 1
        its = list(stats)[:600]
        for it in its:
            k, n, inter = stats[it]
            item = dict(pool["items"].get(it, {"item_content": "", "item_features": ""}), interactors=inter)
            f = featurize(item)
            for j, v in f.items():
                rows.append(r)
                cols.append(j)
                vals.append(v)
            tgt.append(np.log((k + 0.5) / (n - k + 0.5)))
            bid.append(b)
            wts.append(1.0 / len(its))
            r += 1
    X = sp.csr_matrix((vals, (rows, cols)), shape=(r, DIM))
    y = np.array(tgt)
    bid = np.array(bid)
    w = np.array(wts)
    allb = pd.read_parquet(ROOT / "data" / "cache" / "pairs.parquet", columns=["benchmark_id"]).benchmark_id.unique()
    assign = fold_assignment(allb, 7, 0)
    fold = np.array([assign[b] for b in bid])
    import pickle as _pk
    _pk.dump((X, y, bid, w, fold), open(ROOT / "data" / "cache" / "probe_xy.pkl", "wb"))
    Xb = (X != 0).astype(np.float32).tocsc()
    for min_bench in [5, 15]:
        for alpha in [1.0, 10.0]:
            pred = np.zeros(r)
            for f in range(7):
                tr, te = fold != f, fold == f
                # number of distinct training benchmarks in which each feature occurs
                ub = np.unique(bid[tr])
                bidx = {b: i for i, b in enumerate(ub)}
                rows_b = np.array([bidx.get(b, -1) for b in bid])
                Xtr = Xb[tr]
                B = sp.csr_matrix((np.ones(tr.sum()), (rows_b[tr], np.arange(tr.sum()))), shape=(len(ub), tr.sum()))
                bdf = np.asarray(((B @ Xtr) > 0).sum(axis=0)).ravel()
                keep = np.where(bdf >= min_bench)[0]
                Xk = X[:, keep]
                m = Ridge(alpha=alpha, solver="lsqr", fit_intercept=True)
                m.fit(Xk[tr], y[tr], sample_weight=w[tr] * len(ub))
                pred[te] = m.predict(Xk[te])
            df = pd.DataFrame({"b": bid, "y": y, "p": pred})
            bm = df.groupby("b")[["y", "p"]].mean()
            within = df.groupby("b").apply(lambda g: np.corrcoef(g.y, g.p)[0, 1] if g.y.std() > 0 and g.p.std() > 0 else np.nan)
            print(f"min_bench {min_bench} alpha {alpha} feats~{len(keep)}: overall r {np.corrcoef(y, pred)[0, 1]:.3f}  "
                  f"bench-mean r {np.corrcoef(bm.y, bm.p)[0, 1]:.3f}  within-bench r mean {np.nanmean(within):.3f} "
                  f"median {np.nanmedian(within):.3f} frac>0 {np.mean(within.dropna() > 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
