"""PAIEC predictor: nonparametric empirical Bayes over pair accuracy (pure Python, no dependencies).

A revealed label of a subject-benchmark pair is a response on a randomly drawn item, so marginally
it is Bernoulli(q) with q the pair's accuracy. With an empirical prior g(q) over pair accuracies
(fitted on training benchmarks, benchmark-weighted, kernel-smoothed on a grid), the Bayes-optimal
squared-error prediction for an unseen item of the pair is the posterior mean

    E[q | data] with posterior(q) proportional to g(q) * q^(k + w K) * (1 - q)^((n - k) + w (N - K))

where k of n are the pair's own revealed successes and K of N those of OTHER subjects on the same
benchmark (any subject in `labeled` sharing the benchmark_id), down-weighted by w because peers
share the benchmark but not the subject. Optionally the peers' pooled rate also tilts the prior.
"""
import json
import math
import os
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.environ.get("PAIEC_STATE_DIR") or os.path.join(_HERE, "state")
_SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                 "harness_version", "reasoning_effort", "subject_features_extra")


def _load(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


_STATE = _load(os.path.join(_DIR, "eb_prior.json"), {})
_CFG = dict(_load(os.path.join(_HERE, "config.json"), {}))
if os.environ.get("PAIEC_CONFIG"):
    _CFG.update(json.loads(os.environ["PAIEC_CONFIG"]))

W_PEER = float(_CFG.get("w_peer", 0.3))
UNIFORM_MIX = float(_CFG.get("uniform_mix", 0.02))
CLIP = float(_CFG.get("clip", 0.005))
GRID = _STATE.get("grid") or [(i + 0.5) / 100 for i in range(100)]
_prior = _STATE.get("prior") or [1.0] * len(GRID)
_tot = sum(_prior)
PRIOR = [(1 - UNIFORM_MIX) * p / _tot + UNIFORM_MIX / len(GRID) for p in _prior]
LOGQ = [math.log(q) for q in GRID]
LOG1Q = [math.log(1 - q) for q in GRID]
LOGPRIOR = [math.log(p) for p in PRIOR]
PRIOR_MEAN = sum(p * q for p, q in zip(PRIOR, GRID))

_cache = {"obj": None, "len": -1, "counts": None}


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _skey(subject):
    return tuple(_s(subject.get(k, "")) for k in _SUBJECT_KEYS)


def _counts(labeled):
    """{benchmark_id: {subject_key: [successes, n]}} for the current evidence (cached per list object)."""
    if labeled is _cache["obj"] and len(labeled) == _cache["len"]:
        return _cache["counts"]
    counts = {}
    for entry in labeled:
        try:
            (subj, item), y = entry
            y = int(y)
            if y not in (0, 1):
                continue
            c = counts.setdefault(_s(item.get("benchmark_id", "")), {}).setdefault(_skey(subj), [0, 0])
            c[0] += y
            c[1] += 1
        except Exception:
            continue
    _cache.update(obj=labeled, len=len(labeled), counts=counts)
    return counts


def posterior_mean(k, n, kp, np_, w_peer=None):
    w = W_PEER if w_peer is None else w_peer
    a = k + w * kp
    b = (n - k) + w * (np_ - kp)
    if a == 0 and b == 0:
        return PRIOR_MEAN
    lw = [lp + a * l1 + b * l0 for lp, l1, l0 in zip(LOGPRIOR, LOGQ, LOG1Q)]
    mx = max(lw)
    num = den = 0.0
    for q, v in zip(GRID, lw):
        e = math.exp(v - mx)
        num += e * q
        den += e
    return num / den


def predict(input, labeled=None):
    try:
        subject, item = input
        bench = _s((item or {}).get("benchmark_id", ""))
        k = n = kp = np_ = 0
        if labeled and bench:
            per = _counts(labeled).get(bench, {})
            me = _skey(subject or {})
            for sk, (kk, nn) in per.items():
                if sk == me:
                    k, n = kk, nn
                else:
                    kp += kk
                    np_ += nn
        p = posterior_mean(k, n, kp, np_)
    except Exception:
        p = PRIOR_MEAN
    if not math.isfinite(p):
        p = 0.5
    return float(min(1.0 - CLIP, max(CLIP, p)))
