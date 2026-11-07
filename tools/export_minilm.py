"""Export all-MiniLM-L6-v2 weights (float16) and vocab for src/minilm_np.py, then check that the
numpy encoder matches sentence-transformers on pool items.

    python tools/export_minilm.py src/minilm_np_weights
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    bert = st[0].auto_model
    sd = {k: v.detach().cpu().numpy() for k, v in bert.state_dict().items()}
    w = {"emb.word": sd["embeddings.word_embeddings.weight"], "emb.pos": sd["embeddings.position_embeddings.weight"],
         "emb.type": sd["embeddings.token_type_embeddings.weight"], "emb.ln.g": sd["embeddings.LayerNorm.weight"],
         "emb.ln.b": sd["embeddings.LayerNorm.bias"]}
    n_layers = bert.config.num_hidden_layers
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        m = {"q": "attention.self.query", "k": "attention.self.key", "v": "attention.self.value",
             "o": "attention.output.dense", "i": "intermediate.dense", "f": "output.dense"}
        for short, name in m.items():
            w[f"l{i}.{short}.w"] = sd[p + name + ".weight"].T
            w[f"l{i}.{short}.b"] = sd[p + name + ".bias"]
        w[f"l{i}.ln1.g"] = sd[p + "attention.output.LayerNorm.weight"]
        w[f"l{i}.ln1.b"] = sd[p + "attention.output.LayerNorm.bias"]
        w[f"l{i}.ln2.g"] = sd[p + "output.LayerNorm.weight"]
        w[f"l{i}.ln2.b"] = sd[p + "output.LayerNorm.bias"]
    w = {k: v.astype(np.float16) for k, v in w.items()}
    w["n_layers"] = np.array([n_layers])
    w["n_heads"] = np.array([bert.config.num_attention_heads])
    np.savez_compressed(out / "weights.npz", **w)
    vocab = st.tokenizer.get_vocab()
    inv = sorted(vocab.items(), key=lambda kv: kv[1])
    (out / "vocab.txt").write_text("\n".join(t for t, _ in inv) + "\n", encoding="utf-8")
    from emb_text import item_text
    from minilm_np import MiniLM
    enc = MiniLM(str(out))
    texts = []
    for p in sorted((ROOT / "data" / "cache" / "pool").glob("*.pkl"))[::10]:
        pool = pickle.load(open(p, "rb"))
        texts.extend(item_text(it) for it in list(pool["items"].values())[:3])
    ref = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    mine = np.stack([enc.encode_one(t) for t in texts])
    cos = (ref * mine).sum(1)
    print(f"{len(texts)} texts: cosine(ref, numpy) min {cos.min():.5f} mean {cos.mean():.5f}")
    print(f"weights {sum(v.nbytes for v in w.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
