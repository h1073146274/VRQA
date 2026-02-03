#!/usr/bin/env python3
# scripts/01_prepare_data.py
from __future__ import annotations
import argparse
from vrqa.data.prepare_chunks import PrepareConfig, prepare_chunks

def main():
    ap = argparse.ArgumentParser("Prepare VRQA chunks with dual anchors (doc + chunk).")
    ap.add_argument("--input", required=True, help="Raw dataset file: .json or .jsonl (each record is an article).")
    ap.add_argument("--output", required=True, help="Output chunks .jsonl")
    ap.add_argument("--cache_dir", default="cache/anchors", help="Cache dir for anchors to save API cost.")
    ap.add_argument("--backend", default="llm", choices=["llm", "heuristic"], help="Anchor extraction backend.")
    ap.add_argument("--min_chars", type=int, default=800)
    ap.add_argument("--max_chars", type=int, default=2200)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: only process first N docs.")
    args = ap.parse_args()

    cfg = PrepareConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap,
        backend=args.backend,
        sleep_sec=args.sleep,
        max_docs=args.max_docs,
    )
    prepare_chunks(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        cache_dir=args.cache_dir,
    )

if __name__ == "__main__":
    main()
