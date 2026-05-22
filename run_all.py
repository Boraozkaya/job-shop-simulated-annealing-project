from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def _run(script_name: str) -> None:
    print(f"\nRunning {script_name} ...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / script_name)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    _run("run_experiments.py")
    _run("build_presentation_assets.py")
    print("\nAll experiments and presentation assets were regenerated successfully.")


if __name__ == "__main__":
    main()
