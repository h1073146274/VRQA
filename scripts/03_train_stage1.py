#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 training (FiLM only, LinUCB frozen).")
    ap.add_argument("--input", required=True, help="Input chunks JSON or JSONL.")
    ap.add_argument("--output", required=True, help="Output path for QA results.")
    ap.add_argument("--num_rounds", type=int, default=20)
    ap.add_argument("--sample_ratio_per_round", type=float, default=0.1)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--similarity_analysis", default="", help="Optional similarity analysis output path.")
    ap.add_argument("--round_offset", type=int, default=0)
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
        str(args.num_rounds),
        "--sample_ratio_per_round",
        str(args.sample_ratio_per_round),
        "--random_seed",
        str(args.random_seed),
        "--round_offset",
        str(args.round_offset),
        "--enable_film",
        "true",
        "--train_linucb",
        "false",
        "--train_film",
        "true",
    ]
    if args.similarity_analysis:
        cmd.extend(["--similarity_analysis", args.similarity_analysis])
    cmd.extend(extra)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
