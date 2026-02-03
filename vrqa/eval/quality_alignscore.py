#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlignScore-Anchor evaluation script.
- Input 1 (results): chunk_results[].accepted_results[] with question / final_answer / candidate_answer.
- Input 2 (anchors): list with
    id (chunk_id), summary (chunk-level), paragraph_keywords (chunk-level),
    meta.abstract (doc-level), meta.keywords (doc-level).
- Outputs (under --out_dir):
  results_alignscore.csv (per sample)
  summary_alignscore.json / .csv (summary)
  <fig_prefix>_hist.png (histogram)
  <fig_prefix>_scatter.png (scatter)
"""
import argparse
import json
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from alignscore import AlignScore
except Exception:
    print("[ERROR] alignscore package not found. Install it before running.", file=sys.stderr)
    raise


def load_results_file(path: str) -> List[Dict]:
    """
    Load multiple result formats and return a unified list:
      [{"chunk_id":..., "question":..., "answer":...}, ...]
    Supported:
      1) {"chunk_results":[{..., "accepted_results":[...]} , ...]}
      2) List of dicts.
      3) NDJSON/JSON Lines.
      4) Field aliases: answer / generated_answer / final_answer.
    """
    import json

    def _map_row(obj: dict) -> Dict:
        cid = obj.get("chunk_id") or obj.get("chunkId") or obj.get("cid")
        q = (obj.get("question") or obj.get("query") or "").strip()
        a = (obj.get("answer")
             or obj.get("generated_answer")
             or obj.get("final_answer")
             or obj.get("prediction")
             or "").strip()
        if not (q or a):
            return {}
        return {"chunk_id": cid, "question": q, "answer": a}

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # First try single JSON value
    try:
        data = json.loads(raw)
        out: List[Dict] = []

        # Case 1: nested structure
        if isinstance(data, dict) and "chunk_results" in data:
            for ch in data.get("chunk_results", []):
                cid = ch.get("chunk_id")
                rows = ch.get("accepted_results") or ch.get("candidate_results") or []
                for r in rows:
                    mapped = _map_row({
                        "chunk_id": cid,
                        "question": r.get("question"),
                        "answer": (r.get("final_answer")
                                   or r.get("generated_answer")
                                   or r.get("candidate_answer"))
                    })
                    if mapped:
                        out.append(mapped)
            return out

        # Case 2: flat list
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    mapped = _map_row(obj)
                    if mapped:
                        out.append(mapped)
            return out

        # Case 3: top-level question/answer
        if isinstance(data, dict):
            mapped = _map_row(data)
            return [mapped] if mapped else []

    except json.JSONDecodeError:
        # Case 4: NDJSON / JSON Lines
        out: List[Dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mapped = _map_row(obj)
            if mapped:
                out.append(mapped)
        if out:
            return out
        # If still empty, raise
        raise

    # Fallback
    return []



def load_anchor_file(path: str,
                     max_doc_chars: int = 8000,
                     max_chunk_chars: int = 3000) -> Dict[str, Dict[str, str]]:
    """
    Build doc/chunk contexts from the anchors file:
    - Doc: title + abstract + keywords + all chunks for the same articleId (concatenated and truncated).
    - Chunk: summary + paragraph_keywords + content (truncated).
    Note: long contexts are truncated to avoid overly long inputs.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    # 1) Aggregate doc-level context by articleId
    by_article = {}
    for item in arr:
        aid = item.get("articleId") or "NA"
        meta = item.get("meta") or {}
        title = (item.get("title") or "").strip()
        abs_  = (meta.get("abstract") or "").strip()
        dkw   = meta.get("keywords") or []
        # Initialize doc head
        if aid not in by_article:
            head = " ".join(x for x in [title, abs_, " ".join(dkw)] if x).strip()
            by_article[aid] = [head] if head else []

    # Append all chunk content to doc-level text
    for item in arr:
        aid = item.get("articleId") or "NA"
        content = (item.get("content") or "").strip()
        if content:
            by_article[aid].append(content)

    # Final doc text (concat + truncate)
    doc_ctx = {}
    for aid, parts in by_article.items():
        doc_text = "\n".join([p for p in parts if p]).strip()
        if len(doc_text) > max_doc_chars:
            doc_text = doc_text[:max_doc_chars]
        doc_ctx[aid] = doc_text

    # 2) Build chunk/doc anchors per chunk
    idx = {}
    for item in arr:
        cid = item.get("id")
        aid = item.get("articleId") or "NA"
        # Chunk-level anchor
        csum = (item.get("summary") or "").strip()
        ckw  = item.get("paragraph_keywords") or []
        cbody = (item.get("content") or "").strip()
        chunk_anchor = " ".join([x for x in [csum, " ".join(ckw), cbody] if x]).strip()
        if len(chunk_anchor) > max_chunk_chars:
            chunk_anchor = chunk_anchor[:max_chunk_chars]

        # Doc-level anchor
        doc_anchor = doc_ctx.get(aid, "")
        idx[cid] = {"doc_anchor": doc_anchor, "chunk_anchor": chunk_anchor}
    return idx

def batch_alignscore(scorer, contexts: List[str], claims: List[str], batch_size: int = 16) -> List[float]:
    """Compute AlignScore in batches and return scores aligned to claims."""
    assert len(contexts) == len(claims)
    scores = []
    N = len(claims)
    for i in range(0, N, batch_size):
        ctx_batch = contexts[i:i + batch_size]
        clm_batch = claims[i:i + batch_size]
        s = scorer.score(contexts=ctx_batch, claims=clm_batch)
        scores.extend(list(map(float, s)))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True, help="Results JSON (chunk_results[].accepted_results[] supported).")
    ap.add_argument("--anchors_json", required=True, help="Anchors JSON (list with meta/summary/keywords).")
    ap.add_argument("--ckpt_path", required=True, help="Checkpoint path.")
    ap.add_argument("--model", default="", help="Backbone identifier or local path.")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.65, help="Drift threshold (doc and chunk both below).")
    ap.add_argument("--out_csv", default="results_alignscore.csv", help="Per-sample CSV filename or absolute path.")
    ap.add_argument("--fig_prefix", default="fig_alignscore", help="Figure filename prefix.")
    ap.add_argument("--out_dir", default=".", help="Output directory for summaries and figures.")
    args = ap.parse_args()
    if not args.model:
        raise ValueError("--model is required (backbone identifier or local path).")

    # Output directory and path setup
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # out_csv: relative -> join out_dir; absolute -> keep
    out_csv_path = args.out_csv
    if not os.path.isabs(out_csv_path):
        out_csv_path = os.path.join(out_dir, out_csv_path)

    # fig prefix: relative -> join out_dir; absolute -> keep
    fig_prefix = args.fig_prefix
    if not os.path.isabs(fig_prefix):
        fig_prefix = os.path.join(out_dir, fig_prefix)

    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] out_csv: {out_csv_path}")
    print(f"[INFO] fig_prefix: {fig_prefix}")

    # Load data
    triples = load_results_file(args.results_json)
    anchors = load_anchor_file(args.anchors_json)

    if not triples:
        print("[WARN] no question/answer pairs found", file=sys.stderr)

    # Build candidate and anchor inputs
    cand_texts, doc_ctx, chunk_ctx, rows = [], [], [], []
    missing = 0
    for t in triples:
        cid = t["chunk_id"]
        anc = anchors.get(cid)
        if not anc:
            missing += 1
            continue
        qa = (t["question"] + "\n" + t["answer"]).strip()
        cand_texts.append(qa)
        doc_ctx.append(anc["doc_anchor"])
        chunk_ctx.append(anc["chunk_anchor"])
        rows.append({"chunk_id": cid, "question": t["question"], "answer": t["answer"]})
    if missing:
        print(f"[INFO] {missing} samples missing anchors, skipped.")

    # -------- AlignScore --------
    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt_path,
        evaluation_mode="nli_sp",
    )

    # Compute Doc / Chunk scores
    doc_scores = batch_alignscore(scorer, doc_ctx, cand_texts, args.batch_size)
    chunk_scores = batch_alignscore(scorer, chunk_ctx, cand_texts, args.batch_size)

    # Summary and TDR
    df = pd.DataFrame(rows)
    df["align_doc"] = doc_scores
    df["align_chunk"] = chunk_scores
    df["tdr_flag"] = ((df["align_doc"] < args.tau) & (df["align_chunk"] < args.tau)).astype(int)
    tdr = df["tdr_flag"].mean() if len(df) else 0.0

    # Bottleneck alignment: min(doc, chunk) per sample
    df["align_bottleneck"] = df[["align_doc", "align_chunk"]].min(axis=1)

    # Aggregate statistics
    n = len(df)
    mean_doc = float(df["align_doc"].mean()) if n else 0.0
    mean_chunk = float(df["align_chunk"].mean()) if n else 0.0
    med_doc = float(df["align_doc"].median()) if n else 0.0
    med_chunk = float(df["align_chunk"].median()) if n else 0.0
    q_doc = df["align_doc"].quantile([0.1, 0.25, 0.75, 0.9]).to_dict() if n else {}
    q_chunk = df["align_chunk"].quantile([0.1, 0.25, 0.75, 0.9]).to_dict() if n else {}

    mean_bneck = float(df["align_bottleneck"].mean()) if n else 0.0
    med_bneck = float(df["align_bottleneck"].median()) if n else 0.0

    # Print stats
    print(
        f"[STATS] AlignScore-Doc   mean={mean_doc:.4f}  median={med_doc:.4f}  "
        f"p10={q_doc.get(0.1, 0):.4f}  p25={q_doc.get(0.25, 0):.4f}  p75={q_doc.get(0.75, 0):.4f}  p90={q_doc.get(0.9, 0):.4f}"
    )
    print(
        f"[STATS] AlignScore-Chunk mean={mean_chunk:.4f}  median={med_chunk:.4f}  "
        f"p10={q_chunk.get(0.1, 0):.4f}  p25={q_chunk.get(0.25, 0):.4f}  p75={q_chunk.get(0.75, 0):.4f}  p90={q_chunk.get(0.9, 0):.4f}"
    )
    print(f"[STATS] AlignScore-Bottleneck mean={mean_bneck:.4f}  median={med_bneck:.4f}  (min(Doc, Chunk))")
    print(f"[STATS] TDR(tau={args.tau:.2f}) = {tdr:.4f}  (both below threshold)")
    print(f"[STATS] mean(min(Doc,Chunk)) = {mean_bneck:.4f}")

    # Save summary (JSON + single-line CSV)
    summary = {
        "n_samples": n,
        "tau": args.tau,
        "align_doc": {
            "mean": mean_doc,
            "median": med_doc,
            "p10": float(q_doc.get(0.1, 0.0)),
            "p25": float(q_doc.get(0.25, 0.0)),
            "p75": float(q_doc.get(0.75, 0.0)),
            "p90": float(q_doc.get(0.9, 0.0)),
        },
        "align_chunk": {
            "mean": mean_chunk,
            "median": med_chunk,
            "p10": float(q_chunk.get(0.1, 0.0)),
            "p25": float(q_chunk.get(0.25, 0.0)),
            "p75": float(q_chunk.get(0.75, 0.0)),
            "p90": float(q_chunk.get(0.9, 0.0)),
        },
        "align_bottleneck": {"mean": mean_bneck, "median": med_bneck},
        "TDR": tdr,
    }

    with open(os.path.join(out_dir, "summary_alignscore.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {
                "n_samples": n,
                "tau": args.tau,
                "align_doc_mean": mean_doc,
                "align_doc_median": med_doc,
                "align_doc_p10": float(q_doc.get(0.1, 0.0)),
                "align_doc_p25": float(q_doc.get(0.25, 0.0)),
                "align_doc_p75": float(q_doc.get(0.75, 0.0)),
                "align_doc_p90": float(q_doc.get(0.9, 0.0)),
                "align_chunk_mean": mean_chunk,
                "align_chunk_median": med_chunk,
                "align_chunk_p10": float(q_chunk.get(0.1, 0.0)),
                "align_chunk_p25": float(q_chunk.get(0.25, 0.0)),
                "align_chunk_p75": float(q_chunk.get(0.75, 0.0)),
                "align_chunk_p90": float(q_chunk.get(0.9, 0.0)),
                "align_bottleneck_mean": mean_bneck,
                "align_bottleneck_median": med_bneck,
                "TDR": tdr,
            }
        ]
    ).to_csv(os.path.join(out_dir, "summary_alignscore.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] summary written: {os.path.join(out_dir, 'summary_alignscore.(json|csv)')}")

    # Export per-sample CSV
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] per-sample CSV written: {out_csv_path}")
    print(f"[OK] TDR (tau={args.tau:.2f}): {tdr:.3f}  (both below threshold)")

    # Visualization (histogram + scatter)
    plt.figure()
    plt.hist(df["align_doc"], bins=30, alpha=0.7, label="AlignScore-Doc")
    plt.hist(df["align_chunk"], bins=30, alpha=0.7, label="AlignScore-Chunk")
    plt.xlabel("AlignScore")
    plt.ylabel("Count")
    plt.title("AlignScore-Anchor Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_hist.png", dpi=200)

    plt.figure()
    plt.scatter(df["align_doc"], df["align_chunk"], s=8)
    plt.axvline(args.tau, linestyle="--")
    plt.axhline(args.tau, linestyle="--")
    plt.xlabel("AlignScore-Doc")
    plt.ylabel("AlignScore-Chunk")
    plt.title("Doc vs Chunk AlignScore (dashed = tau)")
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_scatter.png", dpi=200)

    print(f"[OK] figures written: {fig_prefix}_hist.png / {fig_prefix}_scatter.png")


if __name__ == "__main__":
    main()
