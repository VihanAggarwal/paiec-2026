"""Rate a stratified sample (N items per benchmark) with a judge model and report how well the ratings
track true item difficulty (mean response over the pool subjects): within-benchmark correlation
(mean over benchmarks) and benchmark-level correlation. Also scores any other ratings file on the
same sample.

    ANTHROPIC_API_KEY=... PAIEC_JUDGE_MODEL=claude-sonnet-5-5 python eval/judge_compare.py 12 out.json [other.json ...]
"""
import json
import math
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import judge  # noqa: E402
from emb_text import item_text  # noqa: E402

n_per, out = int(sys.argv[1]), Path(sys.argv[2])
others = sys.argv[3:]
rng = np.random.default_rng(0)
sample = []  # (bench, text_key, text, item_mean)
for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
    pool = pickle.load(open(p, "rb"))
    st = {}
    for pr in pool["pairs"][:4]:
        for it, _i, y in pr["resp"]:
            s = st.setdefault(it, [0, 0])
            s[0] += y
            s[1] += 1
    its = sorted(st)
    for j in rng.choice(len(its), size=min(n_per, len(its)), replace=False):
        it = its[j]
        t = item_text(pool["items"][it])
        k, n = st[it]
        sample.append((pool["benchmark_id"], judge.text_key(t), t, (k + 0.5) / (n + 1)))
done = json.loads(out.read_text()) if out.exists() else {}
todo = [(k, t) for _, k, t, _ in sample if k not in done]
if todo:
    key = judge._api_key()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, v in zip([k for k, _ in todo], ex.map(lambda kt: judge.call_api(kt[1], key, retries=4), todo)):
            if v is not None:
                done[k] = v
    out.write_text(json.dumps(done))


def score(ratings, name):
    rows = [(b, math.log((ratings[k] + 2) / (102 - ratings[k])), math.log(m / (1 - m)))
            for b, k, _, m in sample if k in ratings]
    by = {}
    for b, z, y in rows:
        by.setdefault(b, []).append((z, y))
    within = [np.corrcoef(*zip(*v))[0, 1] for v in by.values()
              if len(v) >= 5 and np.std([a for a, _ in v]) > 0 and np.std([c for _, c in v]) > 0]
    bm = np.array([(np.mean([a for a, _ in v]), np.mean([c for _, c in v])) for v in by.values()])
    print(f"{name:28s} rated {len(rows)}/{len(sample)}  within-bench r mean {np.mean(within):.3f} "
          f"median {np.median(within):.3f} (n={len(within)})  bench-level r {np.corrcoef(bm[:, 0], bm[:, 1])[0, 1]:.3f}  "
          f"overall r {np.corrcoef([z for _, z, _ in rows], [y for _, _, y in rows])[0, 1]:.3f}")


score(done, out.stem)
for f in others:
    p = Path(f)
    if p.exists():
        score(json.loads(p.read_text()), p.stem)
