"""Sentence-embedding provider for model.py. Never raises: returns None when no encoder is usable.

Order: (1) an in-memory cache, (2) a precomputed cache file named by PAIEC_EMB_CACHE (local
simulation only), (3) the bundled numpy MiniLM encoder (./minilm, see minilm_np.py).
"""
import hashlib
import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emb_text import item_text  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
# Encoder: all-MiniLM-L6-v2 weights bundled in ./minilm and run by minilm_np.py (numpy only).
_mem = {}
_state = {"file_loaded": False, "encoder": None, "encoder_failed": False}


def text_key(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:20]


def _load_file_cache():
    _state["file_loaded"] = True
    path = os.environ.get("PAIEC_EMB_CACHE")
    if not path:
        return
    try:
        import numpy as np
        z = np.load(path)
        for k, v in zip(z["keys"], z["emb"]):
            _mem[str(k)] = v
    except Exception:
        pass


TIME_BUDGET_S = float(os.environ.get("PAIEC_EMB_TIME_BUDGET", "3000"))  # per process, then stop encoding
_state["spent"] = 0.0


try:  # bind at import time: the platform may restore sys.path after importing model.py
    from minilm_np import MiniLM as _MiniLM  # noqa: E402
except Exception:
    _MiniLM = None


class _NumpyEncoder:
    def __init__(self, folder):
        if _MiniLM is None:
            raise RuntimeError("numpy encoder unavailable")
        self.m = _MiniLM(folder)

    def encode_text(self, text):
        return self.m.encode_one(text)


def _encoder():
    if _state["encoder"] is not None or _state["encoder_failed"]:
        return _state["encoder"]
    if os.environ.get("PAIEC_EMB_NO_LIVE"):
        _state["encoder_failed"] = True
        return None
    local = os.path.join(_HERE, "minilm")
    try:
        if os.path.isfile(os.path.join(local, "weights.npz")):
            _state["encoder"] = _NumpyEncoder(local)  # bundled, numpy only
            return _state["encoder"]
    except Exception:
        pass
    _state["encoder_failed"] = True
    return None


def embed(item):
    """L2-normalized embedding (numpy array) of an item dict, or None."""
    try:
        text = item_text(item)
        key = text_key(text)
        v = _mem.get(key)
        if v is not None:
            return v
        if not _state["file_loaded"]:
            _load_file_cache()
            v = _mem.get(key)
            if v is not None:
                return v
        if _state["spent"] > TIME_BUDGET_S:
            return None
        enc = _encoder()
        if enc is None:
            return None
        import time
        t0 = time.time()
        v = enc.encode_text(text)
        _state["spent"] += time.time() - t0
        if len(_mem) > 200000:
            _mem.clear()
        _mem[key] = v
        return v
    except Exception:
        return None
