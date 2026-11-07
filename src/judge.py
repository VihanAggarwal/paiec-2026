"""LLM difficulty judge (Claude Haiku 5.5 via the Anthropic Messages API, Python standard library only).

rate(item) -> float in [0, 100] (the judge's estimate that a typical current AI model succeeds on the
item) or None. Never raises. Sources, in order: in-memory cache, a precomputed cache file named by
PAIEC_JUDGE_CACHE (local simulation), a disk cache in the temp directory, then a live API call.
The API key is read from ANTHROPIC_API_KEY or from anthropic_key.txt next to this file (private
submission builds only; never committed). Live calls stop after a per-process time budget.
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emb_text import item_text  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("PAIEC_JUDGE_MODEL", "claude-haiku-5-5")
API_URL = os.environ.get("PAIEC_JUDGE_URL", "https://api.anthropic.com/v1/messages")  # override: failure tests only
API_VERSION = "2026-10-21"
TIMEOUT_S = float(os.environ.get("PAIEC_JUDGE_TIMEOUT", "10"))
TIME_BUDGET_S = float(os.environ.get("PAIEC_JUDGE_TIME_BUDGET", "1200"))  # per process; with ICL: 6 x (20 + 30) min < 8 h
SYSTEM = ("You are an expert in AI evaluation. You estimate how likely it is that an AI system "
          "succeeds on a single benchmark item.")
PROMPT = ("Below is one item from an AI evaluation benchmark (it may be truncated). Estimate the "
          "probability, from 0 to 100, that a typical current AI model (a modern large language model "
          "or LLM agent) gets a positive outcome on this item, i.e. answers it correctly or succeeds "
          "at the task.\n\nItem:\n<<<\n{text}\n>>>\n\nAnswer with JSON {{\"p\": <integer 0-100>}} and nothing else.")
_INT = re.compile(r"\d{1,3}")

_mem = {}
_failed = {}
_lock = threading.Lock()
_state = {"file_loaded": False, "disk_loaded": False, "spent": 0.0, "key": None, "key_loaded": False,
          "dirty": 0, "disabled": False, "consec_fail": 0}
MAX_CONSEC_FAIL = 8
_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")
_DISK = os.path.join(tempfile.gettempdir(), "paiec_judge_cache.json")


def text_key(text):
    return hashlib.sha1(("judge1\x1f" + text).encode("utf-8", "replace")).hexdigest()[:20]


def _parse_key(raw):
    """Extract an Anthropic key from file bytes or a string (tolerates BOMs, UTF-16, extra text)."""
    if isinstance(raw, bytes):
        text = None
        for enc in (("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"), "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        raw = text or ""
    m = _KEY_RE.search(raw or "")
    return m.group(0) if m else None


def _api_key():
    if not _state["key_loaded"]:
        _state["key_loaded"] = True
        key = _parse_key(os.environ.get("ANTHROPIC_API_KEY", ""))
        if not key:
            try:
                with open(os.path.join(_HERE, "anthropic_key.txt"), "rb") as fh:
                    key = _parse_key(fh.read(4096))
            except Exception:
                key = None
        _state["key"] = key
    return _state["key"]


def _load_caches():
    if not _state["file_loaded"]:
        _state["file_loaded"] = True
        path = os.environ.get("PAIEC_JUDGE_CACHE")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    _mem.update(json.load(fh))
            except Exception:
                pass
    if not _state["disk_loaded"]:
        _state["disk_loaded"] = True
        try:
            with open(_DISK, "r", encoding="utf-8") as fh:
                for k, v in json.load(fh).items():
                    _mem.setdefault(k, v)
        except Exception:
            pass


def _save_disk(force=False):
    if not force and _state["dirty"] < 25:
        return
    try:
        tmp = _DISK + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_mem, fh)
        os.replace(tmp, _DISK)
        _state["dirty"] = 0
    except Exception:
        pass


def call_api(text, key, retries=2):
    """Rating (float) or None. Used by the offline rating tools."""
    return _call(text, key, retries)[0]


def _call(text, key, retries=1, model=None, system=None, raw_prompt=False):
    """(rating or None, status) with status in {"ok", "transient", "permanent"}."""
    model = model or MODEL
    # no thinking: Haiku accepts "disabled" at effort <= high; Sonnet 5.5 takes "between_tools" instead
    thinking = {"type": "between_tools"} if model.startswith("claude-sonnet-5-5") else {"type": "disabled"}
    extra = {"thinking": thinking}
    max_tokens, timeout = 32, TIMEOUT_S
    if model.startswith("claude-opus"):
        extra, max_tokens, timeout = {}, 2048, max(TIMEOUT_S, 25.0)  # Opus 5.5: thinking cannot be disabled
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "system": system or SYSTEM, **extra,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"p": {"type": "integer"}}, "required": ["p"],
            "additionalProperties": False}}},
        "messages": [{"role": "user", "content": text if raw_prompt else PROMPT.format(text=text)}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": key, "anthropic-version": API_VERSION})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            if data.get("stop_reason") == "refusal":
                return None, "ok"
            txt = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
            try:
                val = int(json.loads(txt)["p"])
            except Exception:
                m = _INT.search(txt)
                if not m:
                    return None, "ok"
                val = int(m.group(0))
            return float(min(100, max(0, val))), "ok"
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                return None, "permanent"  # bad key, no credit, unknown model: stop using the judge
            if attempt < retries:
                try:
                    wait = float(exc.headers.get("retry-after", "1"))
                except Exception:
                    wait = 1.0
                time.sleep(min(max(wait, 0.2), 2.0))
                continue
            return None, "transient"
        except (UnicodeError, ValueError):
            return None, "permanent"  # malformed key or header, a retry cannot help
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
                continue
            return None, "transient"
    return None, "transient"


def rate(item, live=True):
    """Cached rating, or a live API call when live=True and the judge is healthy; None otherwise."""
    try:
        text = item_text(item)
        k = text_key(text)
        if k in _mem:
            return _mem[k]
        _load_caches()
        if k in _mem:
            return _mem[k]
        if (not live or k in _failed or _state["disabled"] or os.environ.get("PAIEC_JUDGE_NO_LIVE")
                or _state["spent"] > TIME_BUDGET_S):
            return None
        key = _api_key()
        if not key:
            return None
        t0 = time.time()
        v, status = _call(text, key, retries=1)
        with _lock:
            _state["spent"] += time.time() - t0
            if v is not None:
                _mem[k] = v
                _state["dirty"] += 1
                _state["consec_fail"] = 0
                _save_disk()
            elif status == "ok":
                _failed[k] = 1  # refusal or unparseable answer: do not ask again
                _state["consec_fail"] = 0
            else:
                _failed[k] = 1
                _state["consec_fail"] += 1
                if status == "permanent" or _state["consec_fail"] >= MAX_CONSEC_FAIL:
                    _state["disabled"] = True  # circuit breaker: behave like the judge-free model
        return v
    except Exception:
        return None


def rate_many(items, workers=8, deadline_s=30.0):
    """Rate several items concurrently to warm the cache, giving up after deadline_s seconds."""
    try:
        _load_caches()
        todo, seen = [], set()
        for it in items:
            try:
                k = text_key(item_text(it))
            except Exception:
                continue
            if k in _mem or k in _failed or k in seen:
                continue
            seen.add(k)
            todo.append(it)
        if (not todo or _state["disabled"] or os.environ.get("PAIEC_JUDGE_NO_LIVE") or not _api_key()
                or _state["spent"] > TIME_BUDGET_S):
            return
        from concurrent.futures import ThreadPoolExecutor, wait
        ex = ThreadPoolExecutor(max_workers=workers)
        futs = [ex.submit(rate, it) for it in todo]
        wait(futs, timeout=deadline_s)
        ex.shutdown(wait=False, cancel_futures=True)  # never block on stragglers
        _save_disk(force=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------------------------
# In-context judge: the LLM sees the revealed outcomes on the target's benchmark and the new item.
# The prompt is identical to the one validated by eval/icl_probe.py.
ICL_SYSTEM = ("You are an expert at predicting how AI systems perform on benchmark items. You are shown "
              "revealed outcomes from one benchmark and estimate the probability that a given AI system's "
              "outcome on a new item of the same benchmark is 1.")
ICL_MODEL = os.environ.get("PAIEC_ICL_MODEL", "claude-sonnet-5-5")
ICL_TIME_BUDGET_S = float(os.environ.get("PAIEC_ICL_TIME_BUDGET", "1800"))  # per process
CTX_CHARS = 400
TARGET_CHARS = 1200
_icl_mem = {}


def icl_prompt(subj, own, other, item_text_str, max_own=31, max_other=24):
    """own/other: lists of (system_name, item_text, label). Mirrors eval/icl_probe.build_prompt."""
    own = own[:max_own]
    other = other[:max_other]
    lines = ["Each example is an item from one AI evaluation benchmark with the recorded outcome (1 or 0) "
             "of an AI system on it. Outcome 1 usually means a correct answer or task success, but some "
             "benchmarks define it differently; infer its meaning from the examples.",
             "",
             f"Target AI system: {subj.get('normalized_name', '') or 'unknown'}"
             + (f" (provider: {subj['provider']})" if subj.get("provider") else ""),
             "",
             f"Revealed outcomes of the TARGET system on this benchmark ({len(own)} items, "
             f"{sum(c[2] for c in own)} with outcome 1):"]
    for i, c in enumerate(own, 1):
        lines.append(f"[T{i}] outcome={c[2]} | {c[1]}")
    if not own:
        lines.append("(none yet)")
    lines += ["", f"Revealed outcomes of OTHER systems on this benchmark ({len(other)} items):"]
    for i, c in enumerate(other, 1):
        lines.append(f"[O{i}] system={c[0] or 'unknown'} outcome={c[2]} | {c[1]}")
    if not other:
        lines.append("(none)")
    lines += ["", "New item for the target system:", "<<<", item_text_str, ">>>", "",
              "Estimate the probability (0-100) that the TARGET system's outcome on the new item is 1. "
              "Anchor on the target system's observed rate of outcome 1, shrink toward 50 when it has few "
              "revealed items, and adjust for how this item compares with the examples. "
              'Answer with JSON {"p": <integer 0-100>} and nothing else.']
    return "\n".join(lines)


def icl_rate(subj, own, other, item, model=None):
    """P(outcome = 1) in [0, 1] from the in-context judge, or None (never raises)."""
    model = model or ICL_MODEL
    try:
        if _state["disabled"] or os.environ.get("PAIEC_JUDGE_NO_LIVE") or _state.get("icl_spent", 0.0) > ICL_TIME_BUDGET_S:
            return None
        key = _api_key()
        if not key:
            return None
        prompt = icl_prompt(subj, own, other, item_text(item)[:TARGET_CHARS])
        k = hashlib.sha1((model + "\x1f" + prompt).encode("utf-8", "replace")).hexdigest()[:24]
        if k in _icl_mem:
            return _icl_mem[k]
        t0 = time.time()
        v, status = _call(prompt, key, retries=1, model=model, system=ICL_SYSTEM, raw_prompt=True)
        with _lock:
            _state["icl_spent"] = _state.get("icl_spent", 0.0) + time.time() - t0
            if v is not None:
                _icl_mem[k] = v / 100.0
                _state["consec_fail"] = 0
            elif status != "ok":
                _state["consec_fail"] += 1
                if status == "permanent" or _state["consec_fail"] >= MAX_CONSEC_FAIL:
                    _state["disabled"] = True
        return _icl_mem.get(k)
    except Exception:
        return None
