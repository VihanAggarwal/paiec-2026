"""Dependency-light MiniLM (all-MiniLM-L6-v2) sentence encoder: BERT WordPiece tokenizer + 6-layer
BERT forward pass in numpy, mean pooling, L2 normalization. Weights and vocab are exported once from
the Hugging Face model by tools/export_minilm.py into minilm_np/ (weights.npz, vocab.txt).
Matches sentence-transformers' output to ~1e-3 cosine distance or better (float16 weights).
"""
import math
import os
import unicodedata

import numpy as np

MAX_LEN = 256


def _is_punct(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as fh:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(fh)}
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]

    def _basic(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            if _is_cjk(cp):
                out.append(" " + ch + " ")
            elif ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
                out.append(" ")
            else:
                out.append(ch)
        text = "".join(out).lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        tokens = []
        for word in text.split():
            cur = ""
            for ch in word:
                if _is_punct(ch):
                    if cur:
                        tokens.append(cur)
                        cur = ""
                    tokens.append(ch)
                else:
                    cur += ch
            if cur:
                tokens.append(cur)
        return tokens

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.cls]
        for word in self._basic(text):
            if len(word) > 100:
                ids.append(self.unk)
            else:
                start, sub = 0, []
                while start < len(word):
                    end, piece = len(word), None
                    while start < end:
                        s = word[start:end] if start == 0 else "##" + word[start:end]
                        if s in self.vocab:
                            piece = self.vocab[s]
                            break
                        end -= 1
                    if piece is None:
                        sub = [self.unk]
                        break
                    sub.append(piece)
                    start = end
                ids.extend(sub)
            if len(ids) >= max_len - 1:
                break
        ids = ids[: max_len - 1] + [self.sep]
        return ids


def _layer_norm(x, g, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x):
    # Abramowitz and Stegun 7.1.26, max absolute error 1.5e-7, vectorized
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


class MiniLM:
    def __init__(self, folder):
        z = np.load(os.path.join(folder, "weights.npz"))
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.tok = Tokenizer(os.path.join(folder, "vocab.txt"))
        self.layers = int(self.w["n_layers"][0])
        self.heads = int(self.w["n_heads"][0])

    def encode_one(self, text):
        w = self.w
        ids = np.array(self.tok.encode(text))
        n = len(ids)
        x = w["emb.word"][ids] + w["emb.pos"][:n] + w["emb.type"][0]
        x = _layer_norm(x, w["emb.ln.g"], w["emb.ln.b"])
        d = x.shape[1]
        hd = d // self.heads
        for i in range(self.layers):
            p = f"l{i}."
            q = (x @ w[p + "q.w"] + w[p + "q.b"]).reshape(n, self.heads, hd).transpose(1, 0, 2)
            k = (x @ w[p + "k.w"] + w[p + "k.b"]).reshape(n, self.heads, hd).transpose(1, 0, 2)
            v = (x @ w[p + "v.w"] + w[p + "v.b"]).reshape(n, self.heads, hd).transpose(1, 0, 2)
            a = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            ctx = (a @ v).transpose(1, 0, 2).reshape(n, d)
            x = _layer_norm(x + ctx @ w[p + "o.w"] + w[p + "o.b"], w[p + "ln1.g"], w[p + "ln1.b"])
            h = _gelu(x @ w[p + "i.w"] + w[p + "i.b"])
            x = _layer_norm(x + h @ w[p + "f.w"] + w[p + "f.b"], w[p + "ln2.g"], w[p + "ln2.b"])
        e = x.mean(0)
        return e / (np.linalg.norm(e) + 1e-12)
