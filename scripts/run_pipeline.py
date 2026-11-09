"""End-to-end pipeline.

    python scripts/run_pipeline.py --data          # download Sanger data (~1.3 GB) and build data/processed
    python scripts/run_pipeline.py --experiments   # main (all drugs) + focus panel (~2.5 h on 2 cores)
    python scripts/run_pipeline.py --reports       # tables and figures
    python scripts/run_pipeline.py --notebooks     # execute notebooks/ and learn/
    python scripts/run_pipeline.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


def parallel(experiment, n=2):
    procs = [subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_experiments.py"), experiment,
                               "--part", str(i), "--of", str(n)]) for i in range(n)]
    for p in procs:
        if p.wait() != 0:
            raise SystemExit(f"{experiment} part failed")


def main():
    ap = argparse.ArgumentParser()
    for flag in ["data", "experiments", "reports", "notebooks", "all"]:
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.data or a.all:
        from drugresponse import data
        data.download_all()
        data.process_all()
        from drugresponse import prism
        try:                       # optional: independent-lab test (Broad PRISM, via figshare)
            prism.download()
        except Exception as e:     # noqa: BLE001
            print("PRISM download skipped:", e)
    if a.experiments or a.all:
        parallel("main", a.workers)
        parallel("panel", a.workers)
        run("run_experiments.py", "merge")
    if a.reports or a.all:
        run("make_reports.py")
        run("validate.py")
    if a.notebooks or a.all:
        run("build_notebooks.py")
        run("build_learn.py")


if __name__ == "__main__":
    main()
