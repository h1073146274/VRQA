#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Diversity evaluation runner.")
    ap.add_argument("--method", choices=["semantic", "ragas"], required=True)
    args, extra = ap.parse_known_args()

    module = "vrqa.eval.diversity_semantic" if args.method == "semantic" else "vrqa.eval.diversity_ragas"
    cmd = [sys.executable, "-m", module] + extra
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
