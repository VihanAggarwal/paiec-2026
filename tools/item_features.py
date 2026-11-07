"""Shared, dependency-free item featurizer (hashed sparse features), used by fit scripts and model.py.

Returns a dict {hash_index: value} for an item dict in the official predict() format.
"""
import json
import math
import re
import zlib

DIM = 1 << 18
_WORD = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")


def _h(s):
    return zlib.crc32(s.encode("utf-8", "replace")) & (DIM - 1)


def extract_text(content):
    """Text of an item; multimedia JSON contents are reduced to their text parts plus media tags."""
    tags = []
    if content[:30].lstrip().startswith('{"multimedia_elements"') or content[:2] == '{"':
        try:
            obj = json.loads(content)
            parts = []
            for el in obj.get("multimedia_elements", []) if isinstance(obj, dict) else []:
                ct = str(el.get("content_type", ""))
                tags.append("media:" + ct.split("/")[0])
                if "text" in el:
                    parts.append(str(el["text"]))
            if parts or tags:
                return "\n".join(parts), tags
        except Exception:
            pass
    return content, tags


def featurize(item, max_chars=6000):
    content = item.get("item_content", "") or ""
    feats = item.get("item_features", "") or ""
    inter = item.get("interactors", "") or ""
    text, tags = extract_text(content)
    out = {}

    def add(key, val=1.0):
        j = _h(key)
        out[j] = out.get(j, 0.0) + val

    add("bias")
    for t in tags:
        add(t)
    n = len(text)
    add(f"len:{min(int(math.log2(n + 1)), 20)}")
    lines = text.count("\n") + 1
    add(f"lines:{min(int(math.log2(lines)), 12)}")
    digits = sum(c.isdigit() for c in text[:max_chars])
    add(f"digitfrac:{min(int(10 * digits / max(1, min(n, max_chars))), 9)}")
    nonascii = sum(ord(c) > 127 for c in text[:max_chars])
    add(f"nonascii:{min(int(10 * nonascii / max(1, min(n, max_chars))), 9)}")
    low = text[:max_chars].lower()
    toks = _WORD.findall(low)
    words = {}
    for t in toks:
        words[t] = words.get(t, 0) + 1
    bis = {}
    for a, b in zip(toks, toks[1:]):
        k = a + " " + b
        bis[k] = bis.get(k, 0) + 1
    wsum = 0.0
    tmp = {}
    for t, c in words.items():
        tmp["w:" + t] = 1.0 + math.log(c)
    for t, c in bis.items():
        tmp["b:" + t] = 1.0 + math.log(c)
    for v in tmp.values():
        wsum += v * v
    norm = math.sqrt(wsum) or 1.0
    for k, v in tmp.items():
        add(k, v / norm)
    # first 3 tokens: question type cue ("what", "which", "is", "write", ...)
    for i, t in enumerate(toks[:3]):
        add(f"start{i}:{t}")
    if feats:
        parts = []
        if feats[:1] in "{[":
            try:
                obj = json.loads(feats)
                if isinstance(obj, dict):
                    parts = [f"{k}={v}" for k, v in obj.items()]
            except Exception:
                parts = []
        if not parts:
            parts = [p.strip() for p in re.split(r"[;\n|]", feats) if p.strip()]
        for p in parts[:50]:
            k = p.split("=", 1)[0]
            add("fk:" + k[:100], 0.5)
            add("fv:" + p[:100], 0.5)
    if inter:
        add("inter:" + inter[:100])
    else:
        add("inter:none")
    return out
