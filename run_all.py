"""
run_all.py -- Python orchestrator that mirrors run_all.sh.

This entry point is provided so that the entire pipeline can be
reproduced with a single command on any platform that has Python 3.10+,
without requiring a shell wrapper.

Run
---
::

    python run_all.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    # Inherit PYTHONPATH so that the child process can import our modules
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(cwd)
    if env:
        child_env.update(env)
    res = subprocess.run(cmd, cwd=str(cwd), env=child_env)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
    return res.returncode


def main() -> int:
    # 1. Preprocessing
    run([sys.executable, "-m", "preprocessing.preprocess",
         "--raw-dir", "data/raw",
         "--output", "data/processed_segments.csv",
         "--manifest", "data/segment_manifest.json"])
    # 2. Tests
    run([sys.executable, "tests/test_idm.py"])
    # 3. Calibration
    run([sys.executable, "-m", "idm.calibration",
         "--input", "data/processed_segments.csv",
         "--output", "results/calibration_results.json"])
    # 4. Pareto
    run([sys.executable, "-m", "optimization.multi_objective_optimization",
         "--calibration", "results/calibration_results.json",
         "--output", "results/pareto_front.json"])
    # 5. Closed-loop
    run([sys.executable, "-m", "experiments.closed_loop",
         "--calibration", "results/calibration_results.json",
         "--output", "results/closed_loop_results.json"])
    # 6. Sensitivity
    run([sys.executable, "-m", "analysis.sensitivity",
         "--baseline", "results/pareto_front.json",
         "--output", "results/sensitivity_analysis.json"])
    # 7. VED validation (optional)
    run([sys.executable, "-m", "experiments.ved_validation",
         "--ved-dir", "data/ved_subset",
         "--output", "results/ved_validation.json"])
    # 8. Figures
    run([sys.executable, "-m", "figures.make_figures"])
    # 9. Manuscript
    if shutil.which("tectonic"):
        run(["tectonic", "manuscript.tex"])
    elif shutil.which("pdflatex"):
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"])
        run(["bibtex", "manuscript"])
        run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"])
    else:
        print("No LaTeX compiler available; skipping PDF compilation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
