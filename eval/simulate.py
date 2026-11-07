"""Local evaluator that mimics the official PAIEC streaming protocol on training data.

Leave-benchmarks-out CV: benchmarks are assigned to K folds with a fixed seed; for each fold the
model's optional fit.py is run on the remaining benchmarks only, then model.py is imported with
PAIEC_STATE_DIR pointing at that fold's fitted state, and the held-out benchmarks are scored.

Protocol per held-out snapshot (a stand-in for one formative evaluation):
  * pairs (subject, benchmark) from data/cache/pool, each with 80..400 distinct items; up to
    --max-pairs-per-bench pairs per benchmark; pairs are grouped 1..3 per benchmark and packed into
    snapshots of at most 1000 unique subject-item pairs (acquisition + evaluation), as on Codabench
  * per pair, items are split 50/50 into acquisition / evaluation pools, either per pair
    (--split pair: an item can be evaluation for one subject and acquisition for another) or per
    benchmark (--split benchmark: never)
  * random acquisition order, nested label sets at budgets 0, 1, 3, 7, 15, 31; one recorded response
    is revealed per acquired item; `labeled` holds the revealed labels of EVERY pair in the snapshot
    in [[subject, item], label] form (other subjects on the same benchmark and other benchmarks)
  * predict() is called once per distinct [subject, item] input; Brier over all evaluation
    responses (trials included) per pair and budget; ALC = 0.1 B0 + 0.2 (B1+B3+B7+B15) + 0.1 B31;
    pairs averaged with equal weight. ECE: 10 equal-width bins per pair.

    python eval/simulate.py submissions/v0
    python eval/simulate.py eval/const_model --split benchmark
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import concurrent.futures as cf
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
CAP = 1000
DUMP_PREFIX = os.environ.get("PAIEC_DUMP", "")
DUMP_SNAPS = int(os.environ.get("PAIEC_DUMP_SNAPS", "2"))
DUMP_FROM = int(os.environ.get("PAIEC_DUMP_FROM", "0"))
DUMP_B0 = bool(os.environ.get("PAIEC_DUMP_B0"))
DUMP_ITEMS = int(os.environ.get("PAIEC_DUMP_ITEMS", "25"))
DUMP_CHARS = int(os.environ.get("PAIEC_DUMP_CHARS", "400"))
sys.path.insert(0, str(ROOT / "src"))
MAX_CHUNK = int(os.environ.get("PAIEC_MAX_CHUNK", "3"))  # max pairs of one benchmark per snapshot


def h32(*parts):
    return zlib.crc32("\x1f".join(map(str, parts)).encode())


def fold_assignment(benchmarks, k, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(benchmarks))
    return {b: i % k for i, b in enumerate(order)}


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(p)


def load_model(model_dir, state_dir):
    if state_dir is not None:
        os.environ["PAIEC_STATE_DIR"] = str(state_dir)
    sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location(f"m_{h32(model_dir, state_dir)}", model_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_snapshots(pools, max_pairs, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for bid in sorted(pools):
        pairs = pools[bid]["pairs"][:max_pairs]
        i = 0
        while i < len(pairs):
            k = int(min(len(pairs) - i, rng.integers(1, MAX_CHUNK + 1)))
            chunks.append((bid, list(range(i, i + k))))
            i += k
    rng.shuffle(chunks)
    snaps, cur, size = [], [], 0
    for bid, idxs in chunks:
        n = sum(len({r[0] for r in pools[bid]["pairs"][j]["resp"]}) for j in idxs)
        if cur and size + n > CAP:
            snaps.append(cur)
            cur, size = [], 0
        cur.append((bid, idxs))
        size += n
    if cur:
        snaps.append(cur)
    return snaps


def run_fold(args_tuple):
    model_dir, fold, k, seed, split, max_pairs, max_snaps = args_tuple
    model_dir = Path(model_dir)
    benchmarks = [p.stem for p in (CACHE / "pool").glob("*.pkl")]
    import pandas as pd
    all_bench = pd.read_parquet(CACHE / "pairs.parquet", columns=["benchmark_id"]).benchmark_id.unique()
    assign = fold_assignment(all_bench, k, seed)
    held = sorted(b for b in benchmarks if assign.get(b) == fold)
    state_dir = None
    t0 = time.time()
    if (model_dir / "fit.py").exists():
        state_dir = CACHE / "folds" / model_dir.name / f"k{k}_s{seed}_f{fold}"
        state_dir.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location(f"fit_{fold}", model_dir / "fit.py")
        fit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fit)
        fit.fit(exclude=set(b for b, f in assign.items() if f == fold), out_dir=state_dir, cache_dir=CACHE)
    t_fit = time.time() - t0
    mod = load_model(model_dir, state_dir)
    pools = {}
    for b in held:
        with open(CACHE / "pool" / f"{b}.pkl", "rb") as f:
            pools[b] = pickle.load(f)
    world = os.environ.get("PAIEC_WORLD", "all")
    if world != "all":
        lo, hi = {"hard": (0.0, 0.45), "easy": (0.6, 1.01)}[world]
        keep = {}
        for b, pool in pools.items():
            ys = [y for pr in pool["pairs"][:max_pairs] for _, _, y in pr["resp"]]
            if ys and lo <= sum(ys) / len(ys) < hi:
                keep[b] = pool
        pools = keep
    snaps = build_snapshots(pools, max_pairs, seed * 1000 + fold)
    if max_snaps:
        snaps = snaps[:max_snaps]
    rows, n_calls, t_pred = [], 0, 0.0
    for si, snap in enumerate(snaps):
        rng = np.random.default_rng(h32(seed, fold, si))
        prepared = []
        for bid, idxs in snap:
            pool = pools[bid]
            for j in idxs:
                pr = pool["pairs"][j]
                sid = pr["subject_id"]
                subj = pool["subjects"][sid]
                items = sorted({r[0] for r in pr["resp"]})
                if split == "pair":
                    is_acq = {it: h32("acq", bid, sid, it) % 2 == 0 for it in items}
                else:
                    is_acq = {it: h32("acq", bid, it) % 2 == 0 for it in items}
                acq_items = [it for it in items if is_acq[it]]
                order = [acq_items[i] for i in rng.permutation(len(acq_items))]
                by_item = {}
                for it, inter, y in pr["resp"]:
                    by_item.setdefault(it, []).append((inter, y))
                acq_seq = []
                for it in order[:max(BUDGETS)]:
                    inter, y = by_item[it][int(rng.integers(len(by_item[it])))]
                    item = dict(pool["items"].get(it, {"item_content": "", "item_features": "",
                                                       "benchmark_id": bid}), interactors=inter)
                    item = {"item_content": item["item_content"], "item_features": item["item_features"],
                            "interactors": inter, "benchmark_id": bid}
                    acq_seq.append([[subj, item], int(y)])
                ev = {}
                for it in items:
                    if is_acq[it]:
                        continue
                    for inter, y in by_item[it]:
                        ev.setdefault((it, inter), []).append(y)
                prepared.append((bid, sid, subj, acq_seq, ev, pool))
        base = len(rows)
        dump = DUMP_PREFIX and DUMP_FROM <= si < DUMP_FROM + DUMP_SNAPS
        for bud_i, n in enumerate(BUDGETS):
            labeled = [e for (_, _, _, acq_seq, _, _) in prepared for e in acq_seq[:n]]
            for pi, (bid, sid, subj, acq_seq, ev, pool) in enumerate(prepared):
                ps, ys = [], []
                for ii, ((it, inter), yl) in enumerate(ev.items()):
                    bitem = pool["items"].get(it, {"item_content": "", "item_features": ""})
                    item = {"item_content": bitem["item_content"], "item_features": bitem["item_features"],
                            "interactors": inter, "benchmark_id": bid}
                    t1 = time.time()
                    p = float(mod.predict([subj, item], labeled))
                    t_pred += time.time() - t1
                    n_calls += 1
                    if not (0.0 <= p <= 1.0):
                        raise ValueError(f"invalid probability {p}")
                    ps.extend([p] * len(yl))
                    ys.extend(yl)
                    if dump and (n > 0 or DUMP_B0) and ii < DUMP_ITEMS:
                        from emb_text import item_text
                        ctx = [[(e[0][0] is subj) or (e[0][0] == subj), e[0][0].get("normalized_name", ""),
                                item_text(e[0][1])[:DUMP_CHARS], int(e[1])]
                               for e in labeled if e[0][1].get("benchmark_id") == bid]
                        with open(f"{DUMP_PREFIX}_f{fold}.jsonl", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"fold": fold, "snap": si, "budget": n, "bench": bid, "subject": sid,
                                                 "subj": subj, "item": item_text(item)[:DUMP_CHARS * 3],
                                                 "p_model": p, "ys": [int(v) for v in yl], "ctx": ctx}) + "\n")
                ps, ys = np.array(ps), np.array(ys, dtype=float)
                if bud_i == 0:
                    rows.append({"fold": fold, "snap": si, "benchmark": bid, "subject": sid,
                                 "n_eval_resp": len(ys), "n_eval_items": len({k[0] for k in ev}),
                                 "n_acq": len(acq_seq), "base_rate": float(ys.mean())})
                row = rows[base + pi]
                row[f"B{n}"] = float(np.mean((ps - ys) ** 2))
                row[f"ECE{n}"] = float(ece(ps, ys))
    return {"fold": fold, "rows": rows, "fit_s": t_fit, "n_calls": n_calls, "pred_s": t_pred,
            "n_snaps": len(snaps), "held": held}


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df["ALC"] = sum(w * df[f"B{n}"] for w, n in zip(WEIGHTS, BUDGETS))
    df["ECE_ALC"] = sum(w * df[f"ECE{n}"] for w, n in zip(WEIGHTS, BUDGETS))
    out = {"pairs": int(len(df)), "benchmarks": int(df.benchmark.nunique()),
           "ALC": float(df.ALC.mean()),
           "ALC_se_pairs": float(df.ALC.std(ddof=1) / np.sqrt(len(df))),
           "ALC_se_bench": float(df.groupby("benchmark").ALC.mean().std(ddof=1) / np.sqrt(df.benchmark.nunique())),
           "ALC_bench_weighted": float(df.groupby("benchmark").ALC.mean().mean())}
    for n in BUDGETS:
        out[f"B{n}"] = float(df[f"B{n}"].mean())
    for n in BUDGETS:
        out[f"ECE{n}"] = float(df[f"ECE{n}"].mean())
    out["ECE_ALC"] = float(df.ECE_ALC.mean())
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--eval-folds", default="all", help="comma list, e.g. 0,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["pair", "benchmark"], default="pair")
    ap.add_argument("--max-pairs-per-bench", type=int, default=4)
    ap.add_argument("--max-snaps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    model_dir = Path(a.model_dir).resolve()
    folds = range(a.folds) if a.eval_folds == "all" else [int(x) for x in a.eval_folds.split(",")]
    jobs = [(str(model_dir), f, a.folds, a.seed, a.split, a.max_pairs_per_bench, a.max_snaps) for f in folds]
    t0 = time.time()
    results = []
    with cf.ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
        for r in ex.map(run_fold, jobs):
            results.append(r)
            print(f"fold {r['fold']}: {len(r['rows'])} pairs, {r['n_snaps']} snapshots, fit {r['fit_s']:.0f}s, "
                  f"{r['n_calls']} calls, {1000 * r['pred_s'] / max(1, r['n_calls']):.2f} ms/call", flush=True)
    rows = [x for r in results for x in r["rows"]]
    summ, df = summarize(rows)
    summ.update({"model": model_dir.name, "split": a.split, "folds": a.folds, "eval_folds": list(folds),
                 "seed": a.seed, "max_pairs_per_bench": a.max_pairs_per_bench, "wall_s": time.time() - t0})
    outdir = ROOT / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_dir.name}_{a.split}_k{a.folds}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    df.to_csv(outdir / f"{stem}_pairs.csv", index=False)
    (outdir / f"{stem}.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{model_dir.name} [{a.split} split]: ALC {summ['ALC']:.4f} (se pairs {summ['ALC_se_pairs']:.4f}, "
          f"se bench {summ['ALC_se_bench']:.4f}; bench-weighted {summ['ALC_bench_weighted']:.4f}) "
          f"over {summ['pairs']} pairs / {summ['benchmarks']} benchmarks")
    print("  Brier " + "  ".join(f"B{n}={summ[f'B{n}']:.4f}" for n in BUDGETS))
    print("  ECE   " + "  ".join(f"E{n}={summ[f'ECE{n}']:.4f}" for n in BUDGETS) + f"  ECE_ALC={summ['ECE_ALC']:.4f}")
    print(f"  wall {summ['wall_s']:.0f}s -> {outdir / (stem + '.json')}")


if __name__ == "__main__":
    main()
