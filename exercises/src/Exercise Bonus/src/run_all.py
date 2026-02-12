from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    _run("run_stage_a.py")
    _run("run_stage_b.py")
    _run("run_stage_d.py", "--method", "rag")
    _run("run_stage_c.py")
    _run("run_stage_d.py", "--method", "topic_model")


if __name__ == "__main__":
    main()
