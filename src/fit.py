"""Fit the prior state for model.py from training pairs (data/cache/pairs.parquet).

Writes <out_dir>/prior.json: global prior mean mu0 (benchmark-weighted mean pair logit), variance
starting values tau_b / tau_p, and subject priors: shrunk mean residual (pair logit minus benchmark
mean logit) per lower-cased normalized_name, with a provider-level fallback.

    python submissions/v1/fit.py            # full fit into submissions/v1/state
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KAPPA_NAME = 2.0
KAPPA_PROV = 5.0


def fit(exclude=frozenset(), out_dir=None, cache_dir=None):
    here = Path(__file__).resolve().parent
    cache_dir = Path(cache_dir or here.parents[1] / "data" / "cache")
    out_dir = Path(out_dir or here / "state")
    out_dir.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(cache_dir / "pairs.parquet")
    P = P[~P.benchmark_id.isin(set(exclude)) & (P["items"] >= 20)].copy()
    P["logit"] = np.log((P.k + 0.5) / (P.n - P.k + 0.5))
    bm = P.groupby("benchmark_id").logit.agg(["mean", "var", "size"])
    P["resid"] = P.logit - P.benchmark_id.map(bm["mean"])
    mu0 = float(bm["mean"].mean())
    tau_b = float(np.sqrt(max(bm["mean"].var(), 1e-4)))
    within = bm.loc[bm["size"] >= 3, "var"]
    tau_p = float(np.sqrt(max(within.mean(), 1e-4)))
    P["name"] = P.normalized_name.str.strip().str.lower()
    P["prov"] = P.provider.str.strip().str.lower()
    g = P[P.name != ""].groupby("name").resid.agg(["sum", "size"])
    name = (g["sum"] / (g["size"] + KAPPA_NAME)).round(4).to_dict()
    gp = P[P.prov != ""].groupby("prov").resid.agg(["sum", "size"])
    prov = (gp["sum"] / (gp["size"] + KAPPA_PROV)).round(4).to_dict()
    state = {"mu0": mu0, "tau_b": tau_b, "tau_p": tau_p, "name": name, "provider": prov,
             "n_pairs": int(len(P)), "n_benchmarks": int(len(bm))}
    (out_dir / "prior.json").write_text(json.dumps(state))
    _fit_emb_ridge(set(exclude), out_dir, cache_dir)
    _fit_judge(set(exclude), out_dir, cache_dir)
    return state


JUDGE_FILE = "judge_haiku.json"


def _fit_judge(exclude, out_dir, cache_dir):
    """Slope of the item's mean-response logit on the centered judge logit (rating r in 0..100 ->
    logit((r+2)/(102-r))), fitted on training benchmarks only with equal weight per benchmark.
    Writes <out_dir>/judge_fit.json; skipped if no ratings file exists."""
    import os
    import pickle
    import sys
    path = cache_dir / (os.environ.get("PAIEC_JUDGE_FIT_FILE") or JUDGE_FILE)
    if not path.exists():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from emb_text import item_text
    from judge import text_key
    ratings = json.loads(path.read_text())
    zs, ys, ws = [], [], []
    for p in sorted((cache_dir / "pool").glob("*.pkl")):
        if p.stem in exclude:
            continue
        pool = pickle.load(open(p, "rb"))
        st = {}
        for pr in pool["pairs"][:4]:
            for it, _inter, r in pr["resp"]:
                s = st.setdefault(it, [0, 0])
                s[0] += r
                s[1] += 1
        rows = []
        for it, (k, n) in st.items():
            item = pool["items"].get(it)
            if item is None:
                continue
            rt = ratings.get(text_key(item_text(item)))
            if rt is None:
                continue
            rows.append((np.log((rt + 2.0) / (102.0 - rt)), np.log((k + 0.5) / (n - k + 0.5))))
        for z, y in rows:
            zs.append(z)
            ys.append(y)
            ws.append(1.0 / len(rows))
    if len(zs) < 100:
        return
    z, y, w = np.array(zs), np.array(ys), np.array(ws)
    zc = np.average(z, weights=w)
    yc = np.average(y, weights=w)
    gamma = float(np.sum(w * (z - zc) * (y - yc)) / np.sum(w * (z - zc) ** 2))
    (out_dir / "judge_fit.json").write_text(json.dumps({"gamma": gamma, "center": float(zc), "y_center": float(yc),
                                                         "n_items": int(len(z)), "file": path.name}))


EMB_TAG = "minilm"
EMB_LAMBDA = 10.0


def _fit_emb_ridge(exclude, out_dir, cache_dir):
    """Ridge from item embedding to the item's mean-response logit on training benchmarks (equal
    weight per benchmark). Writes <out_dir>/emb_ridge.json; skipped if no embedding cache exists."""
    import pickle
    path = cache_dir / f"emb_{EMB_TAG}.npz"
    if not path.exists():
        return
    z = np.load(path)
    E = dict(zip(z["keys"], z["emb"]))
    X, y, w = [], [], []
    for p in sorted((cache_dir / "pool").glob("*.pkl")):
        if p.stem in exclude:
            continue
        pool = pickle.load(open(p, "rb"))
        st = {}
        for pr in pool["pairs"]:
            for it, _inter, r in pr["resp"]:
                s = st.setdefault(it, [0, 0])
                s[0] += r
                s[1] += 1
        rows = [(E[pool["benchmark_id"] + "\x1f" + it], np.log((k + 0.5) / (n - k + 0.5)))
                for it, (k, n) in st.items() if pool["benchmark_id"] + "\x1f" + it in E]
        for e, t in rows:
            X.append(e)
            y.append(t)
            w.append(1.0 / len(rows))
    X = np.hstack([np.stack(X), np.ones((len(X), 1))])
    y, w = np.array(y), np.array(w)
    Xw = X * w[:, None]
    beta = np.linalg.solve(X.T @ Xw + EMB_LAMBDA * np.eye(X.shape[1]), Xw.T @ y)
    center = float(np.average(X @ beta, weights=w))
    (out_dir / "emb_ridge.json").write_text(json.dumps({"w_total": [round(float(b), 6) for b in beta],
                                                        "center": center, "lambda": EMB_LAMBDA,
                                                        "model": "sentence-transformers/all-MiniLM-L6-v2"}))


if __name__ == "__main__":
    s = fit()
    print({k: v for k, v in s.items() if not isinstance(v, dict)}, len(s["name"]), "names", len(s["provider"]), "providers")
