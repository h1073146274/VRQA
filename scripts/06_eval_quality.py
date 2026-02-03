#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality evaluation runner.")
    ap.add_argument("--method", choices=["alignscore", "unieval"], required=True)
    args, extra = ap.parse_known_args()

    module = "vrqa.eval.quality_alignscore" if args.method == "alignscore" else "vrqa.eval.quality_unieval"
    cmd = [sys.executable, "-m", module] + extra
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
