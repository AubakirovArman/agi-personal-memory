#!/usr/bin/env python

from pathlib import Path
import subprocess
import sys


def main() -> int:
    target = Path(__file__).with_name("run_path_b_max_gate5_create_bundle.py")
    return subprocess.call([sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
