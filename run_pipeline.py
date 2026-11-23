"""Re-run the whole Hanoi PM2.5 pipeline (or part of it) with one command.

Works on Windows, macOS and Linux. Run from the repository root:

    python run_pipeline.py                  # extract raw cache, run notebooks 01 -> 07 offline
    python run_pipeline.py --from 04        # start at notebook 04 (needs outputs of 01-03)
    python run_pipeline.py --only 07        # run a single notebook
    python run_pipeline.py --regen          # rebuild notebooks from the _make_notebook_*.py scripts first
    python run_pipeline.py --online         # download FIRMS again instead of reusing firms_daily.csv
    python run_pipeline.py --list           # show the steps

Each notebook is executed in its own folder (phase1/ or phase2/) and saved in place with outputs.
Phase 2 reads phase1 data from ../phase1/data/clean/ unless HANOI_AQ_DIR is set.

Note on --regen: the _make_notebook_*.py scripts contain the markdown interpretation
written after the original run (numbers quoted in text). If you change the data or the
parameters, re-run, then read the outputs and update those notes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (id, folder, notebook, generator script, type of work)
STEPS = [
    ("01", "phase1", "01_thu_thap_lam_sach.ipynb", "_make_notebook.py", "data preparation"),
    ("02", "phase1", "02_bien_bo_sung.ipynb", "_make_notebook_02.py", "data preparation"),
    ("03", "phase2", "03_eda.ipynb", "_make_notebook_03.py", "analysis (EDA)"),
    ("04", "phase2", "04_clustering.ipynb", "_make_notebook_04.py", "mining (clustering)"),
    ("05", "phase2", "05_association_rules.ipynb", "_make_notebook_05.py", "mining (association rules)"),
    ("06", "phase2", "06_anomaly.ipynb", "_make_notebook_06.py", "mining (anomaly detection)"),
    ("07", "phase2", "07_sequences.ipynb", "_make_notebook_07.py", "mining (sequential patterns)"),
]

RAW_CACHES = ["raw_openaq_2539_7441.tar.gz", "raw_openmeteo.tar.gz"]


def extract_raw_cache() -> None:
    """Unpack the raw download caches into phase1/data/raw/ (skipped if already there)."""
    data = ROOT / "phase1" / "data"
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in RAW_CACHES:
        archive = data / name
        if not archive.exists():
            print(f"  [skip] {archive} not found")
            continue
        marker = raw / f".extracted_{name}"
        if marker.exists():
            continue
        print(f"  extracting {name} -> phase1/data/raw/")
        with tarfile.open(archive) as tf:
            dest = raw.resolve()
            for m in tf.getmembers():  # refuse paths that escape the target folder
                if not (dest / m.name).resolve().is_relative_to(dest):
                    raise RuntimeError(f"unsafe path in {name}: {m.name}")
            tf.extractall(raw)
        marker.touch()


def regen(folder: str, nb: str, script: str) -> None:
    cwd = ROOT / folder
    print(f"  regenerating {folder}/{nb} from {script}")
    subprocess.run([sys.executable, script, nb], cwd=cwd, check=True)


def execute(folder: str, nb: str, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient

    path = ROOT / folder / nb
    book = nbformat.read(path, as_version=4)
    client = NotebookClient(book, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    nbformat.write(book, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ids = [s[0] for s in STEPS]
    ap.add_argument("--from", dest="start", choices=ids, help="first notebook to run")
    ap.add_argument("--to", dest="end", choices=ids, help="last notebook to run")
    ap.add_argument("--only", choices=ids, help="run just this notebook")
    ap.add_argument("--regen", action="store_true", help="rebuild notebooks from _make_notebook_*.py before running")
    ap.add_argument("--regen-only", action="store_true", help="rebuild notebooks but do not execute them")
    ap.add_argument("--online", action="store_true", help="download FIRMS again (default: reuse firms_daily.csv)")
    ap.add_argument("--timeout", type=int, default=3600, help="max seconds per cell (default 3600)")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()

    if a.list:
        for sid, folder, nb, script, kind in STEPS:
            print(f"{sid}  {folder}/{nb:32s} <- {script:22s} {kind}")
        return 0

    if a.only:
        steps = [s for s in STEPS if s[0] == a.only]
    else:
        lo = ids.index(a.start) if a.start else 0
        hi = ids.index(a.end) if a.end else len(ids) - 1
        steps = STEPS[lo:hi + 1]

    if not a.online:
        os.environ.setdefault("FIRMS_OFFLINE", "1")
    # numba (used by some sklearn/tslearn paths) can fail to cache on network drives
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

    if not a.regen_only and any(s[0] in ("01", "02") for s in steps):
        print("Raw cache:")
        extract_raw_cache()

    for sid, folder, nb, script, kind in steps:
        print(f"\n[{sid}] {folder}/{nb}  ({kind})")
        if a.regen or a.regen_only:
            regen(folder, nb, script)
        if a.regen_only:
            continue
        t0 = time.time()
        execute(folder, nb, a.timeout)
        print(f"  done in {time.time() - t0:.0f} s")
    print("\nAll requested steps finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
