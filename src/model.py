"""PAIEC predictor: hierarchical logistic model with a per-benchmark Laplace GP over revealed labels.

Latent logit of a response of subject s on item i of benchmark b:

    f(s, i) = mu0 + u(s)                       prior mean (fitted on training benchmarks)
            + beta_b                            benchmark offset      ~ N(0, tau_b^2)
            + gamma_{s,b}                       pair offset           ~ N(0, tau_p^2)
            + g_feat(i) + g_text(i)             item-similarity terms, GP with cosine kernels on
                                                item_features tokens and item_content TF-IDF
            + e_i                               exact-item effect     ~ N(0, tau_e^2)

All revealed labels in `labeled` that share the target's benchmark_id (any subject) are used; the
posterior is a Laplace approximation (Rasmussen and Williams, Alg. 3.1) and the predictive
probability uses the probit approximation sigma(m / sqrt(1 + pi v / 8)). With no labels the
prediction is the prior predictive. Pure Python + numpy; deterministic; no network.
"""
import json
import math
import os
import re
import sys
import zlib

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (emb.py)

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small matrices: threads only add contention

try:
    import numpy as np
except Exception:  # pragma: no cover - the pure-Python path below is used instead
    np = None

# Bind sibling modules NOW: the platform may restore sys.path after importing model.py, so a lazy
# `import` inside predict() could fail and silently switch these features off.
try:
    import emb_text as _emb_text_mod
except Exception:
    _emb_text_mod = None
try:
    import emb as _emb_mod
except Exception:
    _emb_mod = None
try:
    import judge as _judge_mod
except Exception:
    _judge_mod = None

_DIR = os.environ.get("PAIEC_STATE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
_SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                 "harness_version", "reasoning_effort", "subject_features_extra")


def _load_json(name, default):
    try:
        with open(os.path.join(_DIR, name), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


_STATE = _load_json("prior.json", {})
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), "r", encoding="utf-8") as _fh:
        _CFG = dict(json.load(_fh))
except Exception:
    _CFG = {}
if os.environ.get("PAIEC_CONFIG"):
    _CFG.update(json.loads(os.environ["PAIEC_CONFIG"]))

MU0 = float(_STATE.get("mu0", 0.2)) + float(_CFG.get("mu0_shift", 0.0))
TAU_B = float(_CFG.get("tau_b", _STATE.get("tau_b", 1.0)))
TAU_P = float(_CFG.get("tau_p", _STATE.get("tau_p", 0.8)))
TAU_F = float(_CFG.get("tau_f", 0.0))
TAU_T = float(_CFG.get("tau_t", 0.0))
TAU_E = float(_CFG.get("tau_e", 0.0))
TAU_M = float(_CFG.get("tau_m", 0.0))   # within-benchmark semantic (sentence-embedding) kernel
MAX_EMB_ENTRIES = int(_CFG.get("max_emb_entries", 400))
JUDGE_SCALE = float(_CFG.get("judge_scale", 0.0))   # multiplier on the fitted judge slope (0 = judge off)
TAU_J = float(_CFG.get("tau_j", 0.0))               # within-benchmark linear kernel on the judge logit
JUDGE_FIT = _load_json("judge_fit.json", {}) if (JUDGE_SCALE or TAU_J) else {}
_J_GAMMA = float(JUDGE_FIT.get("gamma", 0.0))
_J_CENTER = float(JUDGE_FIT.get("center", 0.0))
ICL_WEIGHT = float(_CFG.get("icl_weight", 0.0))   # weight of the in-context LLM judge at budgets >= 1
ICL_B0 = bool(_CFG.get("icl_b0", False))           # also query it with an empty context (budget 0)
ICL_LOW_MODEL = _CFG.get("icl_low_model")            # optional stronger judge when the target has few labels
ICL_LOW_MAX_OWN = int(_CFG.get("icl_low_max_own", 7))
ICL_LOW_WEIGHT = float(_CFG.get("icl_low_weight", ICL_WEIGHT))
if ICL_LOW_MODEL and _judge_mod is not None and _CFG.get("icl_time_budget"):
    _judge_mod.ICL_TIME_BUDGET_S = float(_CFG["icl_time_budget"])
if ICL_WEIGHT and _judge_mod is not None and _CFG.get("icl_model"):
    _judge_mod.ICL_MODEL = str(_CFG["icl_model"])
_icl_ctx = {}
SIGMA_G = float(_CFG.get("sigma_g", 0.0))  # sd of the unknown pool-level mean (0 = off)
TAU_N = float(_CFG.get("tau_n", 0.0))     # near-duplicate kernel exp((cos - 1) / ND_LEN) on raw embeddings
ND_LEN = float(_CFG.get("nd_len", 0.05))
USE_LABELS = bool(_CFG.get("use_labels", True))
USE_PAIR = bool(_CFG.get("use_pair", True))
USE_SUBJECT_PRIOR = bool(_CFG.get("use_subject_prior", True))
SUBJ_SCALE = float(_CFG.get("subject_prior_scale", 1.0))
TEXT_CHARS = int(_CFG.get("text_chars", 4000))
CLIP = float(_CFG.get("clip", 0.005))
NAME_PRIOR = _STATE.get("name", {})
EMB_SCALE = float(_CFG.get("emb_scale", 0.0))      # multiplier on the embedding ridge (total item logit)
EMB_RIDGE = _load_json("emb_ridge.json", {}) if EMB_SCALE != 0.0 else {}
_EMB_W = EMB_RIDGE.get("w_total")
_EMB_C = float(EMB_RIDGE.get("center", 0.0))
PROV_PRIOR = _STATE.get("provider", {})
NEWTON_ITERS = 20
MAX_ENTRIES = int(_CFG.get("max_entries", 1500))
PURE_MAX_ENTRIES = int(_CFG.get("pure_max_entries", 250))
FORCE_PURE = bool(_CFG.get("force_pure", False))
SHARPEN = float(_CFG.get("sharpen", 1.0))
PROBIT_C = float(_CFG.get("probit_c", 1.0))

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_cache = {"t": None}
_solves = {}
_tok_cache = {}


def _sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _subject_key(subject):
    return tuple(_s(subject.get(k, "")) for k in _SUBJECT_KEYS)


def _subject_prior(subject):
    if not USE_SUBJECT_PRIOR:
        return 0.0
    name = _s(subject.get("normalized_name", "")).strip().lower()
    if name in NAME_PRIOR:
        return SUBJ_SCALE * float(NAME_PRIOR[name])
    prov = _s(subject.get("provider", "")).strip().lower()
    return SUBJ_SCALE * float(PROV_PRIOR.get(prov, 0.0))


_emb_off_cache = {}
_global_cache = {}


JUDGE_MODEL = _CFG.get("judge_model")
if (JUDGE_SCALE or TAU_J) and JUDGE_MODEL and _judge_mod is not None:
    _judge_mod.MODEL = str(JUDGE_MODEL)


def _judge_z(item, live=True):
    """Centered judge logit of an item, or 0.0 when no rating is available."""
    if not (JUDGE_SCALE or TAU_J) or _judge_mod is None:
        return 0.0
    try:
        r = _judge_mod.rate(item, live=live)
        if r is None:
            return 0.0
        return math.log((r + 2.0) / (102.0 - r)) - _J_CENTER
    except Exception:
        return 0.0


def _judge_offset(z):
    return JUDGE_SCALE * _J_GAMMA * z


def _global_shift(labeled, bench):
    """Posterior mean and variance of the shift of the pool-level mean, estimated from the revealed
    labels of the OTHER benchmarks in `labeled` (hierarchical: benchmark levels ~ N(mu + shift, tau_b^2),
    shift ~ N(0, SIGMA_G^2)). Each benchmark's mean accuracy is mapped to the latent scale through the
    probit approximation with the within-benchmark variance."""
    if SIGMA_G <= 0 or not labeled:
        return 0.0, 0.0
    fp = (_fingerprint(labeled), bench)
    hit = _global_cache.get(fp)
    if hit is not None:
        return hit
    counts = {}
    for entry in labeled:
        try:
            (_s2, i2), y2 = entry
            b2 = _s(i2.get("benchmark_id", ""))
            if b2 == bench:
                continue
            c = counts.setdefault(b2, [0, 0])
            c[0] += int(y2)
            c[1] += 1
        except Exception:
            continue
    v_within = (TAU_P ** 2 if USE_PAIR else 0.0) + TAU_F ** 2 + TAU_T ** 2 + TAU_M ** 2
    s = math.sqrt(1.0 + math.pi * v_within / 8.0)
    m0 = MU0
    num, prec = 0.0, 1.0 / (SIGMA_G ** 2)
    for k, n in counts.values():
        a = (k + 0.5) / (n + 1.0)
        lev = s * math.log(a / (1.0 - a))
        var = TAU_B ** 2 + s * s / (n * a * (1.0 - a))
        num += (lev - m0) / var
        prec += 1.0 / var
    out = (num / prec, 1.0 / prec if counts else 0.0)
    if len(_global_cache) > 256:
        _global_cache.clear()
    _global_cache[fp] = out
    return out


def _embedding(item):
    if _emb_mod is None:
        return None
    try:
        return _emb_mod.embed(item)
    except Exception:
        return None


def _emb_offset(item):
    """Prior-mean offset from the cross-benchmark embedding ridge (0 when unavailable)."""
    if EMB_SCALE == 0.0 or not _EMB_W:
        return 0.0
    if _emb_mod is None:
        return 0.0
    try:
        v = _emb_mod.embed(item)
        if v is None:
            return 0.0
        w = _EMB_W
        pred = w[-1] + sum(float(a) * b for a, b in zip(v, w[:-1]))
        return EMB_SCALE * (pred - _EMB_C)
    except Exception:
        return 0.0


def _feature_tokens(item):
    feats = _s(item.get("item_features", ""))
    toks = []
    if feats:
        parsed = None
        if feats[:1] in "{[":
            try:
                parsed = json.loads(feats)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                toks.append(f"{k}={v}"[:200])
        else:
            for part in re.split(r"[;\n|]", feats):
                part = part.strip()
                if part:
                    toks.append(part[:200])
    inter = _s(item.get("interactors", ""))
    if inter:
        toks.append("interactors=" + inter[:200])
    return toks


def _text_tokens(item):
    content = _s(item.get("item_content", ""))
    key = (len(content), zlib.crc32(content[:TEXT_CHARS].encode("utf-8", "replace")))
    hit = _tok_cache.get(key)
    if hit is None:
        if len(_tok_cache) > 200000:
            _tok_cache.clear()
        counts = {}
        for t in _TOKEN_RE.findall(content[:TEXT_CHARS].lower()):
            counts[t] = counts.get(t, 0) + 1
        hit = counts
        _tok_cache[key] = hit
    return hit


def _item_key(item):
    c = _s(item.get("item_content", ""))
    f = _s(item.get("item_features", ""))
    return (len(c), zlib.crc32(c.encode("utf-8", "replace")), zlib.crc32(f.encode("utf-8", "replace")))


def _fingerprint(labeled):
    t = _cache.get("t")  # (obj, len, fp) replaced atomically: safe with the in-context judge thread
    if t is not None and labeled is t[0] and len(labeled) == t[1]:
        return t[2]
    h = 0
    for entry in labeled:
        try:
            (subj, item), y = entry
            h = zlib.crc32(repr((_s(item.get("benchmark_id", "")), _subject_key(subj), _item_key(item),
                                 int(y))).encode(), h)
        except Exception:
            continue
    fp = (len(labeled), h)
    _cache["t"] = (labeled, len(labeled), fp)
    return fp


class _Solve:
    """Laplace-approximate GP posterior for one benchmark given its revealed labels."""

    def __init__(self, entries):
        n = len(entries)
        self.n = n
        self.pairs = [e[0] for e in entries]
        self.items = [e[1] for e in entries]
        y = np.array([e[4] for e in entries], dtype=float)
        m = np.array([e[5] for e in entries], dtype=float)
        K = np.full((n, n), TAU_B ** 2)
        if USE_PAIR and TAU_P > 0:
            pa = np.array([hash(p) for p in self.pairs])
            K += (TAU_P ** 2) * (pa[:, None] == pa[None, :])
        if TAU_E > 0:
            ia = np.array([hash(i) for i in self.items])
            K += (TAU_E ** 2) * (ia[:, None] == ia[None, :])
        self.F, self.fvocab, self.fidf = self._matrix([e[2] for e in entries], binary=True) if TAU_F > 0 else (None, None, None)
        self.T, self.tvocab, self.tidf = self._matrix([e[3] for e in entries], binary=False) if TAU_T > 0 else (None, None, None)
        self.M_emb = None
        self.E_raw = None
        if TAU_M > 0:
            embs = [e[6] if len(e) > 6 else None for e in entries]
            dim = next((len(v) for v in embs if v is not None), 0)
            if dim:
                Em = np.array([v if v is not None else np.zeros(dim) for v in embs], dtype=float)
                self.emb_mean = Em.mean(axis=0)
                Ec = Em - self.emb_mean
                nr = np.linalg.norm(Ec, axis=1)
                Ec /= np.where(nr > 0, nr, 1.0)[:, None]
                self.M_emb = Ec
                K += (TAU_M ** 2) * (Ec @ Ec.T)
                if TAU_N > 0:
                    self.E_raw = Em
                    K += (TAU_N ** 2) * np.exp((Em @ Em.T - 1.0) / ND_LEN)
        self.zj = None
        if TAU_J > 0:
            self.zj = np.array([e[7] if len(e) > 7 else 0.0 for e in entries], dtype=float)
            K += (TAU_J ** 2) * np.outer(self.zj, self.zj)
        if self.F is not None:
            K += (TAU_F ** 2) * (self.F @ self.F.T)
        if self.T is not None:
            K += (TAU_T ** 2) * (self.T @ self.T.T)
        K += 1e-6 * np.eye(n)
        f = m.copy()
        eye = np.eye(n)
        for _ in range(NEWTON_ITERS):
            pi = 1.0 / (1.0 + np.exp(-f))
            W = np.maximum(pi * (1 - pi), 1e-10)
            sW = np.sqrt(W)
            B = eye + sW[:, None] * K * sW[None, :]
            b = W * (f - m) + (y - pi)
            a = b - sW * np.linalg.solve(B, sW * (K @ b))
            f_new = m + K @ a
            if np.max(np.abs(f_new - f)) < 1e-6:
                f = f_new
                break
            f = f_new
        pi = 1.0 / (1.0 + np.exp(-f))
        W = np.maximum(pi * (1 - pi), 1e-10)
        sW = np.sqrt(W)
        B = eye + sW[:, None] * K * sW[None, :]
        # predictive variance k_tt - k^T M k with M = S B^-1 S, S = diag(sqrt(W))
        self.M = sW[:, None] * np.linalg.inv(B) * sW[None, :]
        self.resid = y - pi
        self.pair_hash = np.array([hash(p) for p in self.pairs])
        self.item_hash = np.array([hash(i) for i in self.items])

    @staticmethod
    def _matrix(token_lists, binary):
        df = {}
        for toks in token_lists:
            for t in (set(toks) if binary else toks.keys()):
                df[t] = df.get(t, 0) + 1
        n = len(token_lists)
        vocab = {t: j for j, t in enumerate(sorted(t for t, c in df.items() if 2 <= c < n or (n <= 2 and c >= 1)))}
        if not vocab:
            return None, None, None
        idf = np.zeros(len(vocab))
        for t, j in vocab.items():
            idf[j] = math.log((1.0 + n) / (1.0 + df[t])) + 0.1
        X = np.zeros((n, len(vocab)))
        for r, toks in enumerate(token_lists):
            if binary:
                for t in set(toks):
                    j = vocab.get(t)
                    if j is not None:
                        X[r, j] = idf[j]
            else:
                for t, c in toks.items():
                    j = vocab.get(t)
                    if j is not None:
                        X[r, j] = (1.0 + math.log(c)) * idf[j]
        # cosine over the FULL token vector: tokens outside the shared vocabulary still count in the norm
        unseen = math.log((1.0 + n) / 2.0) + 0.1
        norm = np.zeros(n)
        for r, toks in enumerate(token_lists):
            if binary:
                norm[r] = math.sqrt(sum((idf[vocab[t]] if t in vocab else unseen) ** 2 for t in set(toks)))
            else:
                norm[r] = math.sqrt(sum(((1.0 + math.log(c)) * (idf[vocab[t]] if t in vocab else unseen)) ** 2
                                        for t, c in toks.items()))
        X /= np.where(norm > 0, norm, 1.0)[:, None]
        return X, vocab, idf

    def _vec(self, toks, vocab, idf, binary):
        x = np.zeros(len(vocab))
        unseen = math.log(1.0 + self.n) + 0.1
        sq = 0.0
        if binary:
            for t in set(toks):
                j = vocab.get(t)
                w = idf[j] if j is not None else unseen
                if j is not None:
                    x[j] = w
                sq += w * w
        else:
            for t, c in toks.items():
                j = vocab.get(t)
                w = (1.0 + math.log(c)) * (idf[j] if j is not None else unseen)
                if j is not None:
                    x[j] = w
                sq += w * w
        return x / math.sqrt(sq) if sq > 0 else x

    def predict(self, pair, item_key, ftoks, ttoks, m_t, etgt=None, z_t=0.0):
        k = np.full(self.n, TAU_B ** 2)
        if self.zj is not None:
            k += (TAU_J ** 2) * self.zj * z_t
        ktt = (TAU_B ** 2 + (TAU_P ** 2 if USE_PAIR else 0.0) + TAU_E ** 2 + (TAU_M ** 2 if TAU_M > 0 else 0.0)
               + (TAU_N ** 2 if TAU_M > 0 else 0.0) + (TAU_J ** 2) * z_t * z_t)
        if USE_PAIR and TAU_P > 0:
            k += (TAU_P ** 2) * (self.pair_hash == hash(pair))
        if TAU_E > 0:
            k += (TAU_E ** 2) * (self.item_hash == hash(item_key))
        ktt += (TAU_F ** 2 if (TAU_F > 0 and ftoks) else 0.0) + (TAU_T ** 2 if (TAU_T > 0 and ttoks) else 0.0)
        if self.M_emb is not None and etgt is not None:
            v = np.asarray(etgt, dtype=float) - self.emb_mean
            nv = np.linalg.norm(v)
            if nv > 0:
                k += (TAU_M ** 2) * (self.M_emb @ (v / nv))
            if TAU_N > 0:
                k += (TAU_N ** 2) * np.exp((self.E_raw @ np.asarray(etgt, dtype=float) - 1.0) / ND_LEN)
        if self.F is not None and ftoks:
            k += (TAU_F ** 2) * (self.F @ self._vec(ftoks, self.fvocab, self.fidf, True))
        if self.T is not None and ttoks:
            k += (TAU_T ** 2) * (self.T @ self._vec(ttoks, self.tvocab, self.tidf, False))
        mean = m_t + float(k @ self.resid)
        var = max(ktt - float(k @ (self.M @ k)), 1e-9)
        return mean, var


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = A[i][j]
            for k in range(j):
                s -= Li[k] * Lj[k]
            if i == j:
                Li[i] = math.sqrt(max(s, 1e-12))
            else:
                Li[j] = s / Lj[j]
    return L


def _fwd(L, b):
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        Li = L[i]
        s = b[i]
        for k in range(i):
            s -= Li[k] * x[k]
        x[i] = s / Li[i]
    return x


def _bwd(L, b):
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def _sparse_rows(token_lists, binary):
    """Normalized sparse TF-IDF rows (dicts) restricted to tokens shared by >= 2 rows."""
    df = {}
    for toks in token_lists:
        for t in (set(toks) if binary else toks.keys()):
            df[t] = df.get(t, 0) + 1
    n = len(token_lists)
    vocab = {t for t, c in df.items() if 2 <= c < n or (n <= 2 and c >= 1)}
    if not vocab:
        return None, None, n
    idf = {t: math.log((1.0 + n) / (1.0 + df[t])) + 0.1 for t in vocab}
    unseen = math.log((1.0 + n) / 2.0) + 0.1
    rows = []
    for toks in token_lists:
        row, sq = {}, 0.0
        items = ((t, 1) for t in set(toks)) if binary else toks.items()
        for t, c in items:
            w = (1.0 if binary else (1.0 + math.log(c))) * (idf[t] if t in vocab else unseen)
            sq += w * w
            if t in vocab:
                row[t] = w
        nrm = math.sqrt(sq) or 1.0
        rows.append({t: w / nrm for t, w in row.items()})
    return rows, idf, n


class _SolvePy:
    """Same Laplace GP as _Solve, in pure Python (used only if numpy is unavailable)."""

    def __init__(self, entries):
        entries = entries[:PURE_MAX_ENTRIES]
        n = self.n = len(entries)
        self.pairs = [e[0] for e in entries]
        self.items = [e[1] for e in entries]
        y = [float(e[4]) for e in entries]
        m = [float(e[5]) for e in entries]
        K = [[TAU_B ** 2] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if USE_PAIR and TAU_P > 0 and self.pairs[i] == self.pairs[j]:
                    K[i][j] += TAU_P ** 2
                if TAU_E > 0 and self.items[i] == self.items[j]:
                    K[i][j] += TAU_E ** 2
        self.F = self.T = None
        if TAU_F > 0:
            self.F, self.fidf, _ = _sparse_rows([e[2] for e in entries], True)
        if TAU_T > 0:
            self.T, self.tidf, _ = _sparse_rows([e[3] for e in entries], False)
        for rows, tau in ((self.F, TAU_F), (self.T, TAU_T)):
            if rows is None:
                continue
            inv = {}
            for i, r in enumerate(rows):
                for t, w in r.items():
                    inv.setdefault(t, []).append((i, w))
            for lst in inv.values():
                for i, wi in lst:
                    Ki = K[i]
                    for j, wj in lst:
                        Ki[j] += tau * tau * wi * wj
        for i in range(n):
            K[i][i] += 1e-6
        f = list(m)
        for _ in range(NEWTON_ITERS):
            pi = [_sig(v) for v in f]
            W = [max(p * (1 - p), 1e-10) for p in pi]
            sW = [math.sqrt(w) for w in W]
            b = [W[i] * (f[i] - m[i]) + (y[i] - pi[i]) for i in range(n)]
            Kb = [sum(K[i][j] * b[j] for j in range(n)) for i in range(n)]
            B = [[(1.0 if i == j else 0.0) + sW[i] * K[i][j] * sW[j] for j in range(n)] for i in range(n)]
            L = _chol(B)
            t = _bwd(L, _fwd(L, [sW[i] * Kb[i] for i in range(n)]))
            a = [b[i] - sW[i] * t[i] for i in range(n)]
            f_new = [m[i] + sum(K[i][j] * a[j] for j in range(n)) for i in range(n)]
            done = max(abs(f_new[i] - f[i]) for i in range(n)) < 1e-6
            f = f_new
            if done:
                break
        pi = [_sig(v) for v in f]
        self.sW = [math.sqrt(max(p * (1 - p), 1e-10)) for p in pi]
        B = [[(1.0 if i == j else 0.0) + self.sW[i] * K[i][j] * self.sW[j] for j in range(n)] for i in range(n)]
        self.L = _chol(B)
        self.resid = [y[i] - pi[i] for i in range(n)]
        self.inv = []
        for rows in (self.F, self.T):
            idx = {}
            if rows is not None:
                for i, r in enumerate(rows):
                    for t, w in r.items():
                        idx.setdefault(t, []).append((i, w))
            self.inv.append(idx)

    def _target_row(self, toks, idf, binary):
        unseen = math.log(1.0 + self.n) + 0.1
        row, sq = {}, 0.0
        items = ((t, 1) for t in set(toks)) if binary else toks.items()
        for t, c in items:
            w = (1.0 if binary else (1.0 + math.log(c))) * (idf[t] if t in idf else unseen)
            sq += w * w
            if t in idf:
                row[t] = w
        nrm = math.sqrt(sq) or 1.0
        return {t: w / nrm for t, w in row.items()}

    def predict(self, pair, item_key, ftoks, ttoks, m_t):
        n = self.n
        k = [TAU_B ** 2] * n
        ktt = TAU_B ** 2 + (TAU_P ** 2 if USE_PAIR else 0.0) + TAU_E ** 2 + (TAU_M ** 2 if TAU_M > 0 else 0.0)
        ktt += (TAU_F ** 2 if (TAU_F > 0 and ftoks) else 0.0) + (TAU_T ** 2 if (TAU_T > 0 and ttoks) else 0.0)
        for i in range(n):
            if USE_PAIR and TAU_P > 0 and self.pairs[i] == pair:
                k[i] += TAU_P ** 2
            if TAU_E > 0 and self.items[i] == item_key:
                k[i] += TAU_E ** 2
        for rows, idf, toks, tau, binary, inv in ((self.F, getattr(self, "fidf", None), ftoks, TAU_F, True, self.inv[0]),
                                                  (self.T, getattr(self, "tidf", None), ttoks, TAU_T, False, self.inv[1])):
            if rows is None or not toks:
                continue
            for t, w in self._target_row(toks, idf, binary).items():
                for i, wi in inv.get(t, ()):
                    k[i] += tau * tau * w * wi
        mean = m_t + sum(k[i] * self.resid[i] for i in range(n))
        v = _fwd(self.L, [self.sW[i] * k[i] for i in range(n)])
        var = max(ktt - sum(x * x for x in v), 1e-9)
        return mean, var


def _prior_var(ftoks, ttoks):
    # Prior variance of the latent logit for a target with no same-benchmark labels.
    return (TAU_B ** 2 + (TAU_P ** 2 if USE_PAIR else 0.0) + TAU_E ** 2 + (TAU_M ** 2 if TAU_M > 0 else 0.0)
            + (TAU_N ** 2 if TAU_M > 0 else 0.0)
            + (TAU_F ** 2 if (TAU_F > 0 and ftoks) else 0.0) + (TAU_T ** 2 if (TAU_T > 0 and ttoks) else 0.0))


_icl_pool = []


def _icl_executor():
    if not _icl_pool:
        from concurrent.futures import ThreadPoolExecutor
        _icl_pool.append(ThreadPoolExecutor(max_workers=4))
    return _icl_pool[0]


def _predict(subject, item, labeled):
    bench = _s(item.get("benchmark_id", ""))
    skey = _subject_key(subject)
    pair = (skey, bench)
    # Start the in-context judge call now so it overlaps with the item-judge call and the GP solve.
    icl_future = None
    if ICL_WEIGHT > 0 and _judge_mod is not None and bench and (labeled or ICL_B0):
        try:
            icl_future = _icl_executor().submit(_icl, subject, item, labeled, bench, skey)
        except Exception:
            icl_future = None
    g_shift, g_var = _global_shift(labeled, bench) if (USE_LABELS and bench) else (0.0, 0.0)
    z_t = _judge_z(item)
    m_t = MU0 + g_shift + _subject_prior(subject) + _emb_offset(item) + _judge_offset(z_t)
    solve = None
    if USE_LABELS and labeled and bench:
        fp = _fingerprint(labeled)
        key = (fp, bench)
        solve = _solves.get(key)
        if solve is None:
            entries = []
            n_bench = 0
            for entry in labeled:
                try:
                    if _s(entry[0][1].get("benchmark_id", "")) == bench:
                        n_bench += 1
                except Exception:
                    pass
            if JUDGE_SCALE or TAU_J:
                try:
                    _judge_mod.rate_many([e[0][1] for e in labeled
                               if isinstance(e[0][1], dict) and _s(e[0][1].get("benchmark_id", "")) == bench])
                except Exception:
                    pass
            for entry in labeled:
                try:
                    (s2, i2), y2 = entry
                    if _s(i2.get("benchmark_id", "")) != bench:
                        continue
                    y2 = int(y2)
                    if y2 not in (0, 1):
                        continue
                    use_m = TAU_M > 0 and n_bench <= MAX_EMB_ENTRIES
                    zj = _judge_z(i2, live=False)  # just warmed by rate_many; never block here
                    entries.append(((_subject_key(s2), bench), _item_key(i2),
                                    _feature_tokens(i2) if TAU_F > 0 else [],
                                    _text_tokens(i2) if TAU_T > 0 else {},
                                    y2, MU0 + g_shift + _subject_prior(s2) + _emb_offset(i2) + _judge_offset(zj),
                                    _embedding(i2) if use_m else None, zj))
                except Exception:
                    continue
            if len(entries) > MAX_ENTRIES:
                # deterministic subsample keeps the O(n^3) solve bounded on very large benchmarks
                entries.sort(key=lambda e: zlib.crc32(repr((e[0], e[1])).encode()))
                entries = entries[:MAX_ENTRIES]
            solver = _SolvePy if (np is None or FORCE_PURE) else _Solve
            solve = solver(entries) if entries else False
            if len(_solves) > 64:
                _solves.clear()
            _solves[key] = solve
    ftoks = _feature_tokens(item) if TAU_F > 0 else []
    ttoks = _text_tokens(item) if TAU_T > 0 else {}
    if solve:
        if isinstance(solve, _Solve):
            mean, var = solve.predict(pair, _item_key(item), ftoks, ttoks, m_t,
                                      _embedding(item) if TAU_M > 0 else None, z_t)
        else:
            mean, var = solve.predict(pair, _item_key(item), ftoks, ttoks, m_t)
    else:
        mean, var = m_t, _prior_var(ftoks, ttoks) + (TAU_J ** 2) * z_t * z_t
    var += g_var
    p = _sig(SHARPEN * mean / math.sqrt(1.0 + PROBIT_C * math.pi * var / 8.0))
    if icl_future is not None:
        try:
            p_icl = icl_future.result(timeout=65.0 if ICL_LOW_MODEL else 2 * getattr(_judge_mod, "TIMEOUT_S", 10.0) + 5.0)
        except Exception:
            p_icl = None
        if p_icl is not None:
            p_val, w = p_icl
            p = (1.0 - w) * p + w * p_val
    return p


def _icl(subject, item, labeled, bench, skey):
    """In-context judge probability for the target, or None (no labels on the benchmark, or no API)."""
    try:
        key = (_fingerprint(labeled or []), bench)
        ctx = _icl_ctx.get(key)
        if ctx is None:
            from_text = _emb_text_mod.item_text if _emb_text_mod is not None else None
            ctx = []
            for entry in labeled or []:
                try:
                    (s2, i2), y2 = entry
                    if _s(i2.get("benchmark_id", "")) != bench or int(y2) not in (0, 1):
                        continue
                    txt = from_text(i2)[:400] if from_text else _s(i2.get("item_content", ""))[:400]
                    ctx.append((_subject_key(s2), _s(s2.get("normalized_name", "")), txt, int(y2)))
                except Exception:
                    continue
            if len(_icl_ctx) > 64:
                _icl_ctx.clear()
            _icl_ctx[key] = ctx
        if not ctx and not ICL_B0:
            return None
        own = [(c[1], c[2], c[3]) for c in ctx if c[0] == skey]
        other = [(c[1], c[2], c[3]) for c in ctx if c[0] != skey]
        if ICL_LOW_MODEL and 1 <= len(own) <= ICL_LOW_MAX_OWN:
            p = _judge_mod.icl_rate(subject, own, other, item, model=str(ICL_LOW_MODEL))
            return None if p is None else (p, ICL_LOW_WEIGHT)
        p = _judge_mod.icl_rate(subject, own, other, item)
        return None if p is None else (p, ICL_WEIGHT)
    except Exception:
        return None


def predict(input, labeled=None):
    try:
        subject, item = input
        p = _predict(subject or {}, item or {}, labeled or [])
    except Exception:
        p = _sig(MU0)
    if not math.isfinite(p):
        p = 0.5
    return float(min(1.0 - CLIP, max(CLIP, p)))
