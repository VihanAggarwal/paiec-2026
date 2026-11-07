"""Consolidate the downloaded measurement-db tables into compact training files.

Keeps public benchmarks with item-level responses whose values are all 0/1 (response_type
'binary', plus any other benchmark whose responses happen to be strictly binary).

    python tools/build_dataset.py          # reads data/raw, writes data/train/
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "train"
SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "harness_version", "reasoning_effort", "subject_features_extra"]


def _s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench_rows, subj, resp, stats = [], [], [], []
    for bp in sorted(glob.glob(str(RAW / "*" / "formatted_tables" / "benchmarks.parquet"))):
        d = Path(bp).parent
        b = pd.read_parquet(bp).iloc[0]
        if b["release"] != "public" or b["granularity"] != "item":
            continue
        rp = d / "responses.parquet"
        if not rp.exists():
            continue
        r = pd.read_parquet(rp, columns=["subject_id", "item_id", "benchmark_id", "trial", "response"])
        r = r[r["response"].notna()]
        vals = set(np.unique(r["response"].to_numpy()).tolist()) if len(r) else set()
        is_binary = len(r) > 0 and vals <= {0.0, 1.0}
        stats.append({"dir": d.parent.name, "benchmark_id": b["benchmark_id"], "response_type": b["response_type"],
                      "n": len(r), "binary": is_binary})
        if not is_binary:
            continue
        s = pd.read_parquet(d / "subjects.parquet")
        for k in SUBJECT_KEYS:
            s[k] = s[k].map(_s) if k in s else ""
        s["benchmark_id"] = b["benchmark_id"]
        subj.append(s[["subject_id", "benchmark_id"] + SUBJECT_KEYS])
        r = r.copy()
        r["response"] = r["response"].astype(np.int8)
        r["benchmark_id"] = b["benchmark_id"]
        resp.append(r)
        bench_rows.append({k: (json.dumps(list(v)) if isinstance(v, (list, np.ndarray)) else v)
                           for k, v in b.items()})
    S = pd.concat(subj, ignore_index=True)
    R = pd.concat(resp, ignore_index=True)
    Bt = pd.DataFrame(bench_rows)
    S.to_parquet(OUT / "subjects.parquet")
    R.to_parquet(OUT / "responses.parquet")
    Bt.to_parquet(OUT / "benchmarks.parquet")
    pd.DataFrame(stats).to_csv(OUT / "benchmark_selection.csv", index=False)
    print(f"benchmarks kept {len(Bt)} / {len(stats)}; responses {len(R):,}; subject rows {len(S):,}; "
          f"unique subject_id {S.subject_id.nunique():,}; items {R.groupby('benchmark_id').item_id.nunique().sum():,}")


if __name__ == "__main__":
    main()
