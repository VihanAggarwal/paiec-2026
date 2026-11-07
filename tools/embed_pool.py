"""Embed every simulation-pool item with a sentence encoder -> data/cache/emb_<tag>.npz

    python tools/embed_pool.py sentence-transformers/all-MiniLM-L6-v2 minilm
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emb_text import item_text  # noqa: E402

model_name, tag = sys.argv[1], sys.argv[2]
from sentence_transformers import SentenceTransformer  # noqa: E402

m = SentenceTransformer(model_name, device="cpu")
keys, texts = [], []
for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl")):
    pool = pickle.load(open(p, "rb"))
    for iid, it in pool["items"].items():
        keys.append(pool["benchmark_id"] + "\x1f" + iid)
        texts.append(item_text(it))
t0 = time.time()
E = m.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
print(f"{len(texts)} items in {time.time() - t0:.0f}s, dim {E.shape[1]}")
np.savez_compressed(ROOT / "data" / "cache" / f"emb_{tag}.npz", keys=np.array(keys), emb=E.astype(np.float32))
