"""One pass over the raw tables of every kept (strictly binary) benchmark. Writes data/cache/:

  pairs.parquet   per (benchmark, subject_id): responses, positives, distinct items, subject attributes
  pool/<benchmark>.pkl   simulation pool: up to MAX_SUBJ subjects with >= 80 distinct items, each with
                  a random number of items in [80, 400] taken from ONE benchmark-level random item order
                  (so subjects on the same benchmark share items, as in a fixed benchmark item sample),
                  all trials, real subject dicts and item dicts in the official predict() format.

    python tools/build_cache.py
"""
import concurrent.futures as cf
import pickle
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "cache"
SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra"]
MAX_SUBJ = 8
MIN_ITEMS = 80
MAX_CONTENT = 100_000


def _s(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def one(d):
    seed = zlib.crc32(d.encode())
    rng = np.random.default_rng(seed)
    ft = RAW / d / "formatted_tables"
    r = pq.read_table(ft / "responses.parquet",
                      columns=["subject_id", "item_id", "trial", "interactors", "response"]).to_pandas()
    r = r[r.response.notna()]
    bid = pq.read_table(ft / "benchmarks.parquet").to_pandas().iloc[0]["benchmark_id"]
    subj = pd.read_parquet(ft / "subjects.parquet")
    for k in SUBJECT_KEYS:
        subj[k] = subj[k].map(_s) if k in subj else ""
    subj = subj.drop_duplicates("subject_id").set_index("subject_id")
    g = r.groupby("subject_id")
    pairs = pd.DataFrame({"n": g.size(), "k": g.response.sum(), "items": g.item_id.nunique()}).reset_index()
    pairs["benchmark_id"] = bid
    for k in SUBJECT_KEYS:
        pairs[k] = pairs.subject_id.map(subj[k]).fillna("")
    elig = pairs.loc[pairs["items"] >= MIN_ITEMS, "subject_id"].tolist()
    pool = None
    if elig:
        chosen = sorted(rng.choice(elig, size=min(MAX_SUBJ, len(elig)), replace=False).tolist())
        all_items = r.item_id.unique()
        order = {it: i for i, it in enumerate(rng.permutation(all_items))}
        rr = r[r.subject_id.isin(chosen)].copy()
        rr["ord"] = rr.item_id.map(order)
        pool_pairs, need = [], set()
        for sid in chosen:
            x = rr[rr.subject_id == sid]
            its = x.drop_duplicates("item_id").sort_values("ord").item_id.tolist()
            n_target = int(rng.integers(MIN_ITEMS, 401))
            its = its[:max(MIN_ITEMS, n_target)]
            x = x[x.item_id.isin(set(its))]
            need.update(its)
            pool_pairs.append({"subject_id": sid,
                               "resp": list(zip(x.item_id.tolist(), x.interactors.map(_s).tolist(),
                                                x.response.astype(int).tolist()))})
        it = pq.read_table(ft / "items.parquet", columns=["item_id", "content", "item_features"],
                           filters=pc.field("item_id").isin(list(need))).to_pandas()
        items = {row.item_id: {"item_content": _s(row.content)[:MAX_CONTENT],
                               "item_features": _s(row.item_features),
                               "benchmark_id": bid}
                 for row in it.itertuples()}
        subjects = {sid: {k: subj.loc[sid, k] for k in SUBJECT_KEYS} for sid in chosen}
        pool = {"benchmark_id": bid, "n_items_total": int(len(all_items)), "subjects": subjects,
                "items": items, "pairs": pool_pairs}
        with open(OUT / "pool" / f"{bid}.pkl", "wb") as f:
            pickle.dump(pool, f, protocol=4)
    return pairs, bid, (len(pool["pairs"]) if pool else 0)


def main():
    (OUT / "pool").mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(ROOT / "data" / "train" / "benchmark_selection.csv")
    dirs = sel.loc[sel.binary, "dir"].tolist()
    allp = []
    with cf.ProcessPoolExecutor(4) as ex:
        for pairs, bid, npool in ex.map(one, dirs):
            allp.append(pairs)
    P = pd.concat(allp, ignore_index=True)
    P.to_parquet(OUT / "pairs.parquet")
    print(f"pairs {len(P):,}; pool files {len(list((OUT / 'pool').glob('*.pkl')))}")


if __name__ == "__main__":
    main()
