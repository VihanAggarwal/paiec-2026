"""In-context LLM judge probe on dumped evaluation episodes (eval/simulate.py with PAIEC_DUMP).

For each dumped (pair, budget, evaluation item) the LLM sees the revealed outcomes on the same
benchmark (the target system's first, then other systems') and the new item, and returns P(outcome=1).
Compares Brier with the model's own prediction (p_model) on identical items, and a logistic blend
fitted on one half of the benchmarks and evaluated on the other half (and vice versa).

    ANTHROPIC_API_KEY=... python eval/icl_probe.py data/cache/icl/dump data/cache/icl/sonnet.json [--model ...]
"""
import argparse
import glob
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import judge  # noqa: E402

W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
SYSTEM = ("You are an expert at predicting how AI systems perform on benchmark items. You are shown "
          "revealed outcomes from one benchmark and estimate the probability that a given AI system's "
          "outcome on a new item of the same benchmark is 1.")


def build_prompt(rec, max_own=31, max_other=24, variant="base"):
    subj = rec["subj"]
    own = [c for c in rec["ctx"] if c[0]][:max_own]
    other = [c for c in rec["ctx"] if not c[0]][:max_other]
    lines = ["Each example is an item from one AI evaluation benchmark with the recorded outcome (1 or 0) "
             "of an AI system on it. Outcome 1 usually means a correct answer or task success, but some "
             "benchmarks define it differently; infer its meaning from the examples.",
             "",
             f"Target AI system: {subj.get('normalized_name', '') or 'unknown'}"
             + (f" (provider: {subj['provider']})" if subj.get("provider") else ""),
             "",
             f"Revealed outcomes of the TARGET system on this benchmark ({len(own)} items, "
             f"{sum(c[3] for c in own)} with outcome 1):"]
    for i, c in enumerate(own, 1):
        lines.append(f"[T{i}] outcome={c[3]} | {c[2]}")
    if not own:
        lines.append("(none yet)")
    lines += ["", f"Revealed outcomes of OTHER systems on this benchmark ({len(other)} items):"]
    for i, c in enumerate(other, 1):
        lines.append(f"[O{i}] system={c[1] or 'unknown'} outcome={c[3]} | {c[2]}")
    if not other:
        lines.append("(none)")
    if variant == "hint":
        lines += ["", "A statistical model that pools all revealed outcomes estimates the probability that the "
                      f"target system's outcome on the new item is 1 at {round(100 * rec['p_model'])}%. It is "
                      "usually well calibrated but cannot read the items; use it as a prior and correct it "
                      "with what you see."]
    lines += ["", "New item for the target system:", "<<<", rec["item"], ">>>", "",
              "Estimate the probability (0-100) that the TARGET system's outcome on the new item is 1. "
              "Anchor on the target system's observed rate of outcome 1, shrink toward 50 when it has few "
              "revealed items, and adjust for how this item compares with the examples. "
              'Answer with JSON {"p": <integer 0-100>} and nothing else.']
    return "\n".join(lines)


def call(prompt, key, model):
    import urllib.request
    import urllib.error
    import time
    thinking = {"type": "between_tools"} if model.startswith("claude-sonnet-5-5") else {"type": "disabled"}
    extra = {"thinking": thinking}
    max_tokens = 32
    if model.startswith("claude-opus"):
        extra, max_tokens = {}, 2048  # thinking cannot be disabled on Opus 5.5; low effort keeps it short
    body = json.dumps({"model": model, "max_tokens": max_tokens, "system": SYSTEM, **extra,
                       "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {
                           "type": "object", "properties": {"p": {"type": "integer"}}, "required": ["p"],
                           "additionalProperties": False}}},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(judge.API_URL, data=body, method="POST", headers={
                "content-type": "application/json", "x-api-key": key, "anthropic-version": judge.API_VERSION})
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.load(resp)
            txt = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
            return float(min(100, max(0, int(json.loads(txt)["p"]))))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504, 529):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1)
    return None


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def fit_blend(X, y, w):
    """Weighted logistic regression by Newton (few parameters)."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (w * (y - p)) - 1e-3 * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-call", action="store_true")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--budgets", default="1,3,7,15,31")
    a = ap.parse_args()
    keep_b = {int(x) for x in a.budgets.split(",")}
    recs = [json.loads(l) for f in sorted(glob.glob(a.dump + "_f*.jsonl")) for l in open(f)]
    recs = [r for r in recs if r["budget"] in keep_b]
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    prompts = {}
    for r in recs:
        pr = build_prompt(r, variant=a.variant)
        k = hashlib.sha1((a.model + "\x1f" + pr).encode()).hexdigest()[:24]
        r["k"] = k
        prompts[k] = pr
    todo = [k for k in prompts if k not in done]
    print(f"{len(recs)} records, {len(prompts)} distinct prompts, {len(todo)} to call; "
          f"mean prompt chars {np.mean([len(p) for p in prompts.values()]):.0f}", flush=True)
    if todo and not a.no_call:
        key = judge._api_key()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, (k, v) in enumerate(zip(todo, ex.map(lambda kk: call(prompts[kk], key, a.model), todo)), 1):
                if v is not None:
                    done[k] = v
                if i % 500 == 0:
                    out.write_text(json.dumps(done))
                    print(f"  {i}/{len(todo)}", flush=True)
        out.write_text(json.dumps(done))
    rows = [r for r in recs if r["k"] in done]
    print(f"rated {len(rows)}/{len(recs)}")
    # per (pair, budget) Brier, then ALC-style weighting over budgets 1..31 (budget 0 is unchanged)
    benches = sorted({r["bench"] for r in rows})
    half = {b: i % 2 for i, b in enumerate(benches)}
    feats = lambda r: [1.0, logit(r["p_model"]), logit(done[r["k"]] / 100.0)]
    preds = {}
    for h in (0, 1):
        tr = [r for r in rows if half[r["bench"]] != h]
        X = np.array([feats(r) for r in tr for _ in r["ys"]])
        y = np.array([yy for r in tr for yy in r["ys"]], dtype=float)
        w = np.ones(len(y))
        beta = fit_blend(X, y, w)
        print(f"  blend fitted on half {1 - h}: intercept {beta[0]:+.2f}, model {beta[1]:.2f}, llm {beta[2]:.2f}")
        for r in rows:
            if half[r["bench"]] == h:
                preds[id(r)] = 1 / (1 + math.exp(-float(np.dot(beta, feats(r)))))
    agg = {}
    for r in rows:
        pm, pl, pb = r["p_model"], done[r["k"]] / 100.0, preds[id(r)]
        key = (r["fold"], r["snap"], r["bench"], r["subject"], r["budget"])
        a_ = agg.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
        for yy in r["ys"]:
            a_[0] += (pm - yy) ** 2
            a_[1] += (pl - yy) ** 2
            a_[2] += (pb - yy) ** 2
            a_[3] += (0.5 * (pm + pl) - yy) ** 2
            a_[4] += 1
    per_budget = {}
    for (f, s, b, sj, n), v in agg.items():
        per_budget.setdefault(n, []).append([v[i] / v[4] for i in range(4)])
    tot = np.zeros(4)
    for n in sorted(per_budget):
        m = np.mean(per_budget[n], axis=0)
        tot += W[n] * m
        print(f"budget {n:2d}: Brier model {m[0]:.4f}  llm {m[1]:.4f}  blend {m[2]:.4f}  avg {m[3]:.4f}  "
              f"({len(per_budget[n])} pairs)")
    print(f"weighted over budgets 1-31 (ALC weights, budget 0 excluded): model {tot[0]:.4f}  llm {tot[1]:.4f}  "
          f"blend {tot[2]:.4f}  avg {tot[3]:.4f}   -> ALC change of blend: {tot[2] - tot[0]:+.4f}")


if __name__ == "__main__":
    main()
