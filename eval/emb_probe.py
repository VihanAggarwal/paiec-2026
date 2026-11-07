"""Leave-benchmarks-out probe: do sentence embeddings predict (a) within-benchmark item difficulty
and (b) benchmark level on UNSEEN benchmarks? Same 7 folds as eval/simulate.py.

Targets per pool item: smoothed logit of its mean response over the pool subjects (all trials);
benchmark level = mean item logit; residual = item logit - benchmark level.

    python eval/emb_probe.py minilm
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from simulate import fold_assignment  # noqa: E402


def targets():
    rows = []
    for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
        pool = pickle.load(open(p, "rb"))
        b = pool["benchmark_id"]
        st = {}
        for pr in pool["pairs"]:
            for it, inter, y in pr["resp"]:
                s = st.setdefault(it, [0, 0])
                s[0] += y
                s[1] += 1
        for it, (k, n) in st.items():
            rows.append((b + "\x1f" + it, b, np.log((k + 0.5) / (n - k + 0.5)), n))
    return pd.DataFrame(rows, columns=["key", "b", "logit", "n"])


def ridge(X, y, w, lam):
    Xw = X * w[:, None]
    A = X.T @ Xw + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, Xw.T @ y)


def main():
    tag = sys.argv[1]
    z = np.load(ROOT / "data" / "cache" / f"emb_{tag}.npz")
    E = dict(zip(z["keys"], z["emb"]))
    df = targets()
    df = df[df.key.isin(E)].reset_index(drop=True)
    X = np.stack([E[k] for k in df.key])
    X = np.hstack([X, np.ones((len(X), 1))])
    df["lvl"] = df.groupby("b").logit.transform("mean")
    df["res"] = df.logit - df.lvl
    df["w"] = 1.0 / df.groupby("b").b.transform("size")
    allb = pd.read_parquet(ROOT / "data" / "cache" / "pairs.parquet", columns=["benchmark_id"]).benchmark_id.unique()
    assign = fold_assignment(allb, 7, 0)
    fold = df.b.map(assign).to_numpy()
    # benchmark-level design: mean embedding of the benchmark's items
    bm = df.groupby("b").apply(lambda g: X[g.index].mean(axis=0))
    BX = np.stack(bm.values)
    blvl = df.groupby("b").lvl.first().loc[bm.index].to_numpy()
    bfold = np.array([assign[b] for b in bm.index])
    for lam in [1.0, 10.0, 100.0]:
        pr = np.zeros(len(df))
        pb = np.zeros(len(bm))
        for f in range(7):
            tr, te = fold != f, fold == f
            beta = ridge(X[tr], df.res.to_numpy()[tr], df.w.to_numpy()[tr], lam)
            pr[te] = X[te] @ beta
            btr, bte = bfold != f, bfold == f
            bb = ridge(BX[btr], blvl[btr], np.ones(btr.sum()), lam / 10)
            pb[bte] = BX[bte] @ bb
        df["pr"] = pr
        within = df.groupby("b").apply(lambda g: np.corrcoef(g.res, g.pr)[0, 1] if g.pr.std() > 0 and g.res.std() > 0 else np.nan)
        # how much of the within-benchmark residual variance a calibrated slope would explain
        slope = np.polyfit(df.pr, df.res, 1)[0]
        r2 = 1 - np.average((df.res - slope * df.pr) ** 2, weights=df.w) / np.average(df.res ** 2, weights=df.w)
        # single item-level model of the TOTAL logit (level + residual), benchmark-equal weights
        pt = np.zeros(len(df))
        for f in range(7):
            tr, te = fold != f, fold == f
            bt = ridge(X[tr], df.logit.to_numpy()[tr], df.w.to_numpy()[tr], lam)
            pt[te] = X[te] @ bt
        df["pt"] = pt
        wt = df.groupby("b").apply(lambda g: np.corrcoef(g.res, g.pt)[0, 1] if g.pt.std() > 0 else np.nan)
        bt_r = np.corrcoef(df.groupby("b").lvl.first(), df.groupby("b").pt.mean())[0, 1]
        print(f"   total-logit model: item r {np.corrcoef(df.logit, pt)[0, 1]:.3f}, within-bench r mean {np.nanmean(wt):.3f}, "
              f"bench-mean r {bt_r:.3f}")
        print(f"lam {lam:6.1f}: item residual r (pooled) {np.corrcoef(df.res, pr)[0, 1]:.3f}, within-bench r mean "
              f"{np.nanmean(within):.3f} median {np.nanmedian(within):.3f} frac>0 {np.mean(within.dropna() > 0):.2f}, "
              f"slope {slope:.2f}, R2 {r2:.3f} | bench level r {np.corrcoef(blvl, pb)[0, 1]:.3f} "
              f"(rmse {np.sqrt(np.mean((blvl - pb) ** 2)):.2f} vs sd {blvl.std():.2f})", flush=True)


if __name__ == "__main__":
    main()
