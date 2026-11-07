"""Local smoke test for a submission directory or ZIP (no network, synthetic data).

Mimics the platform's streaming protocol: several subject-benchmark pairs, nested label sets at
budgets 0, 1, 3, 7, 15, 31, where `labeled` holds the revealed labels of ALL sampled pairs (other
subjects on the same benchmark and other benchmarks), in [[subject, item], label] form.

Checks: import from a foreign working directory, finite float in [0, 1], determinism, no mutation
of the arguments, per-call latency, robustness to empty/odd field values.

    python tools/run_smoke_test.py submissions/v1
    python tools/run_smoke_test.py dist/v1.zip
"""
import copy
import importlib.util
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from pathlib import Path

BUDGETS = (0, 1, 3, 7, 15, 31)


def _subject(i):
    return {
        "normalized_name": f"smoke-model-{i}",
        "provider": ["OpenAI", "Anthropic", "Google", "", "Meta"][i % 5],
        "release_date": ["2024-05-13", "2025-02-01", "", "2023-11-06", "not-a-date"][i % 5],
        "access_date": ["2025-11-01", "", "2026-01-15", "2024-12-01", "2025-06-30"][i % 5],
        "harness": ["", "default", "react-agent", "lm-eval-harness", "x"][i % 5],
        "harness_version": ["", "1.0", "v0.4.2", "", ""][i % 5],
        "reasoning_effort": ["", "high", "medium", "low", ""][i % 5],
        "subject_features_extra": ["", '{"temperature": 0}', "", "k=v;a=b", ""][i % 5],
    }


def _item(b, j):
    return {
        "item_content": ["", "What is 2+2?", "Write a function that reverses a string.\n" * (j % 7),
                         "Unicode: é中文 \U0001f600", "x" * (50 * j)][j % 5],
        "item_features": ["", "tier=medium;points=1", '{"difficulty": "hard"}', "a", ""][j % 5],
        "interactors": ["", "python", "", "browser,terminal", ""][j % 5],
        "benchmark_id": f"benchmark_{b:06d}",
    }


def _load(sub_dir: Path):
    # Same as the official checker's loader: the submission dir is on sys.path ONLY during import.
    previous = list(sys.path)
    sys.path.insert(0, str(sub_dir))
    try:
        spec = importlib.util.spec_from_file_location("smoke_submission_model", sub_dir / "model.py")
        mod = importlib.util.module_from_spec(spec)
        t0 = time.time()
        spec.loader.exec_module(mod)
        print(f"import model.py: {time.time() - t0:.2f}s")
    finally:
        sys.path[:] = previous
    for name in ("judge", "emb"):
        m = getattr(mod, f"_{name}_mod", None)
        print(f"  sibling module {name}: {'loaded' if m is not None else 'MISSING'}")
    return mod


def _check(value, where):
    if isinstance(value, bool):
        raise AssertionError(f"{where}: returned bool")
    v = float(value)
    if not math.isfinite(v) or v < 0 or v > 1:
        raise AssertionError(f"{where}: invalid probability {value!r}")
    return v


def run(sub_dir: Path):
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as elsewhere:
        os.chdir(elsewhere)  # the platform does not promise cwd == submission dir
        try:
            mod = _load(sub_dir)
            predict = mod.predict
            rng = random.Random(0)
            # 3 benchmarks x 4 subjects = 12 pairs, 40 eval items + 40 acquisition items per pair.
            n_items = 6 if QUICK else 40
            pairs = [(s, b) for b in range(3 if not QUICK else 1) for s in range(4 if not QUICK else 2)]
            acq = {p: [_item(p[1], 1000 + 37 * p[0] + j) for j in range(40)] for p in pairs}
            evl = {p: [_item(p[1], j) for j in range(n_items)] for p in pairs}
            truth = {p: rng.random() for p in pairs}
            labels = {p: [int(rng.random() < truth[p]) for _ in range(31)] for p in pairs}
            n_calls, worst, total = 0, 0.0, 0.0
            briers = {}
            for budget in BUDGETS:
                labeled = [[[_subject(p[0]), acq[p][k]], labels[p][k]] for p in pairs for k in range(budget)]
                frozen = copy.deepcopy(labeled)
                sq, n = 0.0, 0
                for p in pairs:
                    for it in evl[p]:
                        inp = [_subject(p[0]), it]
                        inp_frozen = copy.deepcopy(inp)
                        t0 = time.time()
                        v = _check(predict(inp, labeled), f"budget {budget} pair {p}")
                        dt = time.time() - t0
                        worst, total, n_calls = max(worst, dt), total + dt, n_calls + 1
                        if inp != inp_frozen or labeled != frozen:
                            raise AssertionError("predict() mutated its arguments")
                        y = int(rng.random() < truth[p])
                        sq += (v - y) ** 2
                        n += 1
                # determinism: repeat one call on a deep copy
                p = pairs[0]
                a = predict([_subject(p[0]), evl[p][0]], copy.deepcopy(labeled))
                b = predict([_subject(p[0]), evl[p][0]], copy.deepcopy(labeled))
                if float(a) != float(b):
                    raise AssertionError("predict() is not deterministic")
                briers[budget] = sq / n
                print(f"budget {budget:2d}: {len(labeled):4d} labels in `labeled`, synthetic Brier {sq / n:.4f}")
            # labeled=None and an item without benchmark_id must not crash at budget 0
            _check(predict([_subject(0), {k: v for k, v in _item(0, 1).items() if k != "benchmark_id"}], []),
                   "item without benchmark_id")
            _check(predict([_subject(1), _item(1, 2)], None), "labeled=None")
            print(f"{n_calls} calls, mean {1000 * total / n_calls:.2f} ms, worst {1000 * worst:.1f} ms")
        finally:
            os.chdir(old_cwd)
    print("SMOKE TEST PASSED")


QUICK = False


def main():
    global QUICK
    if "--quick" in sys.argv:
        sys.argv.remove("--quick")
        QUICK = True  # 2 pairs x 6 items: keeps live LLM calls cheap
    if "--no-numpy" in sys.argv:
        sys.argv.remove("--no-numpy")
        sys.modules["numpy"] = None  # any `import numpy` now raises ImportError
        print("numpy blocked for this run")
    target = Path(sys.argv[1]).resolve()
    if target.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(target) as zf:
                zf.extractall(tmp)
            run(Path(tmp))
    else:
        run(target)


if __name__ == "__main__":
    main()
