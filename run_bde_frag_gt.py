#!/usr/bin/env python3
"""Run BDGen molecular optimization through the shared six-strategy interface."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "BDGen" / "run_bdgen.py"), run_name="__main__")
