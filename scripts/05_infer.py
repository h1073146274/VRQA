#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference-only run.")
    ap.add_argument("--input", required=True, help="Input chunks JSON or JSONL.")
    ap.add_argument("--output", required=True, help="Output path for QA results.")
    ap.add_argument("--similarity_analysis", default="", help="Optional similarity analysis output path.")
    ap.add_argument("--random_seed", type=int, default=42)
    args, extra = ap.parse_known_args()

    cmd = [
        sys.executable,
        "-m",
        "vrqa.pipeline.generate",
        "--input",
        args.input,
        "--output",
        args.output,
        "--num_rounds",
        "1",
        "--sample_ratio_per_round",
        "1.0",
        "--random_seed",
        str(args.random_seed),
        "--ucb_exploit_only",
        "true",
    ]
    if args.similarity_analysis:
        cmd.extend(["--similarity_analysis", args.similarity_analysis])
    cmd.extend(extra)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
