"""Materialize submissions/<name>/ from src/ with a given config, fitted on ALL training benchmarks.

    python tools/make_submission.py v3 '{"use_pair": true}'
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make(name, config):
    out = ROOT / "submissions" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "state").mkdir(parents=True)
    for f in ("model.py", "fit.py"):
        shutil.copy(ROOT / "src" / f, out / f)
    for f in (ROOT / "src").glob("*.py"):
        if f.name not in ("model.py", "fit.py"):
            shutil.copy(f, out / f.name)
    if config.get("tau_m", 0) or config.get("emb_scale", 0):
        shutil.copytree(ROOT / "src" / "minilm", out / "minilm")  # bundled numpy sentence encoder
    (out / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True))
    spec = importlib.util.spec_from_file_location("fit_full", out / "fit.py")
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    fit.fit(exclude=set(), out_dir=out / "state", cache_dir=ROOT / "data" / "cache")
    return out


if __name__ == "__main__":
    print(make(sys.argv[1], json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")))
