"""Zip the CONTENTS of submissions/<version>/ (model.py at the ZIP root) into dist/<version>.zip.

    python tools/build_zip.py v0
"""
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc"}
SKIP_NAMES = {"fit.py"}  # training-time only; not needed by the platform


def build(version: str) -> Path:
    src = ROOT / "submissions" / version
    if not (src / "model.py").exists():
        raise SystemExit(f"{src}/model.py not found")
    out = ROOT / "dist" / f"{version}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.is_file()
                   and not (set(p.relative_to(src).parts) & SKIP_DIRS)
                   and p.suffix not in SKIP_SUFFIXES and p.name not in SKIP_NAMES)
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        for p in files:
            info = ZipInfo(p.relative_to(src).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, p.read_bytes())
    return out


if __name__ == "__main__":
    for v in sys.argv[1:] or ["v0"]:
        path = build(v)
        print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
