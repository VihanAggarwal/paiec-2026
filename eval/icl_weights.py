"""Per-budget blend weight w(n) for p = (1-w) p_model + w p_llm, chosen on one half of the benchmarks
and evaluated on the other (and vice versa), on the dumped probe episodes.

    python eval/icl_weights.py data/cache/icl/sonnet.json [model]
"""
import glob
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, "eval")
sys.path.insert(0, "src")
from icl_probe import build_prompt  # noqa: E402

W = {1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
GRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
done = json.load(open(sys.argv[1]))
model = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-5-5"
recs = [json.loads(l) for f in sorted(glob.glob("data/cache/icl/dump_f*.jsonl")) for l in open(f)]
rows = []
for r in recs:
    k = hashlib.sha1((model + "\x1f" + build_prompt(r)).encode()).hexdigest()[:24]
    rows.append((r, done.get(k)))
benches = sorted({r["bench"] for r, _ in rows})
half = {b: i % 2 for i, b in enumerate(benches)}


def pair_brier(sel, wfun):
    agg = {}
    for r, pl in sel:
        w = wfun(r["budget"]) if pl is not None else 0.0
        p = (1 - w) * r["p_model"] + w * ((pl or 0) / 100.0)
        key = (r["fold"], r["snap"], r["bench"], r["subject"])
        for y in r["ys"]:
            a = agg.setdefault(key, {}).setdefault(r["budget"], [0.0, 0])
            a[0] += (p - y) ** 2
            a[1] += 1
    return {k: sum(W[b] * v[0] / v[1] for b, v in d.items()) for k, d in agg.items()}


out = {}
chosen = {}
for h in (0, 1):
    tr = [x for x in rows if half[x[0]["bench"]] != h]
    best = {}
    for n in W:
        sub = [x for x in tr if x[0]["budget"] == n]
        scores = {w: np.mean(list(pair_brier(sub, lambda b, w=w: w).values())) for w in GRID}
        best[n] = min(scores, key=scores.get)
    chosen[h] = best
    te = [x for x in rows if half[x[0]["bench"]] == h]
    out.update(pair_brier(te, lambda b: best[b]))
base = pair_brier(rows, lambda b: 0.0)
avg = pair_brier(rows, lambda b: 0.5)
d_cv = np.array([out[k] - base[k] for k in base])
d_avg = np.array([avg[k] - base[k] for k in base])
print("weights chosen per half:", chosen)
print(f"fixed 0.5: {d_avg.mean():+.4f} (se {d_avg.std(ddof=1) / np.sqrt(len(d_avg)):.4f});  "
      f"per-budget (cross-fitted): {d_cv.mean():+.4f} (se {d_cv.std(ddof=1) / np.sqrt(len(d_cv)):.4f})")
