"""Re-key data/cache/emb_<tag>.npz (benchmark + item_id) by the hash of the encoder text, for the
model's PAIEC_EMB_CACHE lookup during local simulation.

    python tools/emb_bytext.py minilm
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emb import text_key  # noqa: E402
from emb_text import item_text  # noqa: E402

tag = sys.argv[1]
z = np.load(ROOT / "data" / "cache" / f"emb_{tag}.npz")
E = dict(zip(z["keys"], z["emb"]))
keys, embs = [], []
for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
    pool = pickle.load(open(p, "rb"))
    for iid, it in pool["items"].items():
        k = pool["benchmark_id"] + "\x1f" + iid
        if k in E:
            keys.append(text_key(item_text(it)))
            embs.append(E[k])
np.savez(ROOT / "data" / "cache" / f"emb_{tag}_bytext.npz", keys=np.array(keys), emb=np.stack(embs))
print(len(keys), "keys,", len(set(keys)), "distinct texts")
