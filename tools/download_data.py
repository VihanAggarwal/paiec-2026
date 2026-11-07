"""Download the measurement-db tables needed for training into paiec/data/.

Needs a Hugging Face token for an account approved for the gated dataset (HF_TOKEN env var or
`huggingface-cli login`). Only the small tabular files are fetched (benchmarks, subjects,
responses, items); traces and assets are skipped.

    python tools/download_data.py                 # tables except items
    python tools/download_data.py --items         # also items.parquet (large)
"""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--items", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    sha = api.repo_info(REPO, repo_type="dataset").sha
    top = [e.path for e in api.list_repo_tree(REPO, repo_type="dataset", revision=sha)
           if not hasattr(e, "size") or e.__class__.__name__ == "RepoFolder"]
    print(f"revision {sha}, {len(top)} top-level folders")

    def files_of(d):
        try:
            return [e.path for e in api.list_repo_tree(REPO, path_in_repo=f"{d}/formatted_tables",
                                                        repo_type="dataset", revision=sha)
                    if e.path.endswith(".parquet")]
        except Exception:
            return []

    with cf.ThreadPoolExecutor(args.workers) as ex:
        listing = dict(zip(top, ex.map(files_of, top)))
    wanted = {"benchmarks.parquet", "subjects.parquet", "responses.parquet", "response.parquet"}
    if args.items:
        wanted.add("items.parquet")
    todo = [f for fs in listing.values() for f in fs if f.rsplit("/", 1)[-1] in wanted]
    print(f"{sum(1 for v in listing.values() if v)} folders with formatted_tables, {len(todo)} files to fetch")

    def get(f):
        for attempt in range(4):
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", revision=sha, local_dir=out)
            except Exception as exc:  # retry transient network errors
                if attempt == 3:
                    print("FAILED", f, type(exc).__name__)
                    return None
                time.sleep(2 ** attempt)

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "manifest.json").write_text(json.dumps({"repo": REPO, "revision": sha, "files": sorted(todo)}, indent=1))


if __name__ == "__main__":
    main()
