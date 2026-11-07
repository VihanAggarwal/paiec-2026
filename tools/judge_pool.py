"""Rate simulation-pool items with the LLM judge (src/judge.py) -> data/cache/judge_haiku.json
{text_key: rating}. Resumable; needs ANTHROPIC_API_KEY in the environment.

    ANTHROPIC_API_KEY=... python tools/judge_pool.py --pairs 4 --workers 24 [--limit N]
"""
import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import judge  # noqa: E402
from emb_text import item_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "judge_haiku.json"))
    a = ap.parse_args()
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    texts = {}
    for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
        pool = pickle.load(open(p, "rb"))
        for pr in pool["pairs"][: a.pairs]:
            for it in sorted({r[0] for r in pr["resp"]}):
                item = pool["items"].get(it)
                if item is None:
                    continue
                t = item_text(item)
                texts.setdefault(judge.text_key(t), t)
    todo = [(k, t) for k, t in texts.items() if k not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(texts)} distinct item texts, {len(done)} already rated, {len(todo)} to rate", flush=True)
    key = judge._api_key()
    t0, n_ok, n_fail = time.time(), 0, 0

    def one(kt):
        k, t = kt
        return k, judge.call_api(t, key, retries=4)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (k, v) in enumerate(ex.map(one, todo), 1):
            if v is None:
                n_fail += 1
            else:
                done[k] = v
                n_ok += 1
            if i % 1000 == 0 or i == len(todo):
                out.write_text(json.dumps(done))
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  ok {n_ok} fail {n_fail}  {rate:.1f}/s", flush=True)
    out.write_text(json.dumps(done))


if __name__ == "__main__":
    main()
