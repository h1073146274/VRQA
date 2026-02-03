#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diversity and semantic evaluation utilities."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


EPS = 1e-12


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required. Install: pip install torch")


def _looks_like_path(s: str) -> bool:
    if not s:
        return False
    return os.path.exists(s) or ("/" in s) or ("\\" in s)


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def get_by_path(d: Dict[str, Any], key_path: str) -> Any:
    if not key_path:
        return None
    cur: Any = d
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def has_by_path(d: Dict[str, Any], key_path: str) -> bool:
    return get_by_path(d, key_path) is not None


def tokenize(text: str, mode: str = "nltk") -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    if mode == "whitespace":
        return text.split()
    if mode == "regex":
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if mode == "nltk":
        try:
            from nltk.tokenize import TreebankWordTokenizer
            return TreebankWordTokenizer().tokenize(text)
        except Exception:
            return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    raise ValueError(f"Unknown tokenizer mode: {mode}")


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


_EN_PUNC = re.compile(r"""[\s,.;:!?'"`~@#$%^&*_\-+=/\\|<>\[\](){}]""")

_EN_Q_PREFIX = re.compile(
    r"^(according to (the )?(passage|text)|based on (the )?(passage|text)|"
    r"which of the following|following statements?|from the passage|"
    r"as stated in the passage|in the context of the passage|"
    r"choose the (best|correct) answer|read the passage and answer)"
    r".{0,30}?:?\s*", re.I
)
_EN_STOP = set("""
a an the is are was were be been being am do does did doing have has had having
of to in for on with by from at as that this these those it its their there here
and or but if while so because than then also very really just not no nor only
can could may might must shall should will would into out up down over under
about above below across between among before after during without within upon
what which who whom whose where when why how such other another same different
""".split())


def _is_likely_en(text: str) -> bool:
    if not text:
        return False
    ascii_cnt = sum(1 for ch in text if ord(ch) < 128)
    return (ascii_cnt / max(1, len(text))) > 0.8


def _strip_template_prefix(q: str, lang: str) -> str:
    q = q or ""
    q = _EN_Q_PREFIX.sub("", q)
    q = _EN_PUNC.sub(" ", q)
    return re.sub(r"\s+", " ", q).strip()


def _token_seq_for_distinct(text: str, *, lang: str, token_unit: str, norm: str) -> List[str]:
    text = text or ""
    if norm == "raw":
        s = text
    elif norm == "strip_template":
        s = _strip_template_prefix(text, lang)
    else:
        s = _strip_template_prefix(text, lang)

    if token_unit == "word":
        s2 = _EN_PUNC.sub(" ", s)
        toks = [w for w in re.split(r"\s+", s2) if w]
        if norm == "content":
            toks = [w for w in toks if w.lower() not in _EN_STOP]
        return toks
    return [ch for ch in s if not ch.isspace()]


def distinct_n_macro(groups: List[List[str]], n: int, *, lang: str, token_unit: str, norm: str) -> float:
    vals = []
    for sents in groups:
        token_lists = [_token_seq_for_distinct(s, lang=lang, token_unit=token_unit, norm=norm) for s in sents]
        grams = []
        for toks in token_lists:
            grams.extend(ngrams(toks, n))
        total = len(grams)
        vals.append(0.0 if total == 0 else len(set(grams)) / total)
    return float(np.mean(vals)) if vals else 0.0


def distinct_n_micro(sentences: List[str], n: int, *, lang: str, token_unit: str, norm: str) -> float:
    grams = []
    for s in sentences:
        toks = _token_seq_for_distinct(s, lang=lang, token_unit=token_unit, norm=norm)
        grams.extend(ngrams(toks, n))
    total = len(grams)
    return (len(set(grams)) / total) if total else 0.0


@dataclass
class SimCSEEncoder:
    model_name: str
    device: str = "cpu"
    batch_size: int = 64
    max_length: int = 128

    def __post_init__(self):
        _require_torch()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> "torch.Tensor":
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_emb = outputs.pooler_output
            else:
                batch_emb = outputs.last_hidden_state[:, 0, :]
            embs.append(batch_emb)
        return torch.cat(embs, dim=0)


def avg_pairwise_cosine(emb: "torch.Tensor") -> float:
    _require_torch()
    if emb.shape[0] < 2:
        return float("nan")
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.T
    n = sim.shape[0]
    triu = sim.triu(diagonal=1)
    denom = n * (n - 1) / 2
    return float(triu.sum().item() / max(denom, 1.0))


def _cosine_kernel_01(emb: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.T
    K = (sim + 1.0) * 0.5
    n = K.shape[0]
    K = K + torch.eye(n, device=K.device, dtype=K.dtype) * 1e-6
    return K


def vendi_score_embed(emb: "torch.Tensor", q: float | str = 1.0) -> float:
    _require_torch()
    if emb.shape[0] < 2:
        return float("nan")

    K = _cosine_kernel_01(emb)
    lam = torch.linalg.eigvalsh(K).clamp_min(0.0)
    s = lam.sum().clamp_min(EPS)
    p = (lam / s).clamp_min(EPS)

    if isinstance(q, str) and q.lower() == "inf":
        return float((1.0 / p.max()).item())

    qv = float(q)
    if abs(qv - 1.0) < 1e-8:
        H = -(p * p.log()).sum()
        return float(torch.exp(H).item())
    return float((p.pow(qv).sum().clamp_min(EPS).pow(1.0 / (1.0 - qv))).item())


def diversity_at_k_farthest_first(emb: "torch.Tensor", k: int, *, seed: int = 42) -> Tuple[float, int]:
    _require_torch()
    n = emb.shape[0]
    if k <= 0 or n < 2:
        return float("nan"), min(n, max(k, 0))
    k = min(k, n)

    emb = F.normalize(emb, p=2, dim=1)
    sim = (emb @ emb.T).detach()
    dist = (1.0 - sim).clamp_min(0.0)

    avg_d = dist.mean(dim=1)
    start = int(torch.argmax(avg_d).item())
    selected = [start]
    remaining = set(range(n))
    remaining.remove(start)

    while len(selected) < k and remaining:
        sel_idx = torch.tensor(selected, device=dist.device, dtype=torch.long)
        rem_idx = torch.tensor(sorted(remaining), device=dist.device, dtype=torch.long)
        d_min = dist[rem_idx][:, sel_idx].min(dim=1).values
        best_pos = int(torch.argmax(d_min).item())
        chosen = int(rem_idx[best_pos].item())
        selected.append(chosen)
        remaining.remove(chosen)

    if len(selected) < 2:
        return 0.0, len(selected)

    sel = torch.tensor(selected, device=dist.device, dtype=torch.long)
    sel_emb = emb[sel]
    avg_sim = avg_pairwise_cosine(sel_emb)
    return float(1.0 - avg_sim) if not math.isnan(avg_sim) else float("nan"), len(selected)


def self_bertscore(groups: List[List[str]], device: str, bertscore_model: str, *, lang: str = "en") -> Tuple[float, str]:
    from bert_score import score as bert_score

    hyp_list: List[str] = []
    ref_list: List[List[str]] = []
    for sents in groups:
        if len(sents) < 2:
            continue
        for i in range(len(sents)):
            hyp_list.append(sents[i])
            ref_list.append(sents[:i] + sents[i + 1:])

    if not hyp_list:
        return float("nan"), ""

    def _run(model_type: str) -> float:
        _, _, F1 = bert_score(
            cands=hyp_list,
            refs=ref_list,
            lang=lang,
            rescale_with_baseline=False,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        return float(F1.mean().item())

    if _looks_like_path(bertscore_model):
        base = os.path.basename(os.path.normpath(bertscore_model))
        try:
            return _run(base), base
        except Exception:
            return _run(bertscore_model), bertscore_model

    return _run(bertscore_model), bertscore_model


def self_bleu_sacrebleu(groups: List[List[str]], max_ngram_order: int) -> float:
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="13a", smooth_method="exp", effective_order=True, max_ngram_order=max_ngram_order)
    scores = []
    for sents in groups:
        if len(sents) < 2:
            continue
        for i in range(len(sents)):
            hyp = sents[i]
            refs = sents[:i] + sents[i + 1:]
            sb = bleu.sentence_score(hypothesis=hyp, references=refs)
            scores.append(float(sb.score))
    return float(np.mean(scores)) if scores else float("nan")


def self_bleu_coco(groups: List[List[str]], max_ngram_order: int) -> float:
    from nlgeval.pycocoevalcap.bleu.bleu import Bleu
    hyps = {}
    refs = {}
    idx = 0
    for sents in groups:
        if len(sents) < 2:
            continue
        for i in range(len(sents)):
            hyps[idx] = [sents[i]]
            refs[idx] = sents[:i] + sents[i + 1:]
            idx += 1
    if idx == 0:
        return float("nan")
    scorer = Bleu(4)
    score, _ = scorer.compute_score(refs, hyps)
    return float(score[max_ngram_order - 1])


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_json_or_jsonl(path: str) -> Any:
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _autodetect_first_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if has_by_path(d, k):
            return k
    return None


def load_chunk_resources(
    chunks_path: str,
    *,
    chunk_file_chunk_id_key: str = "id",
    chunk_file_doc_id_key: str = "",
) -> Dict[str, Any]:
    if not chunks_path:
        return {"chunk_to_doc": {}, "inferred_keys": {}}

    obj = load_json_or_jsonl(chunks_path)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        if "chunks" in obj and isinstance(obj["chunks"], list):
            items = obj["chunks"]
        else:
            items = list(obj.values())
    else:
        raise ValueError(f"Unsupported chunks file format: {type(obj)}")

    inferred = {}
    first_dict = next((x for x in items if isinstance(x, dict)), None) or {}
    if not chunk_file_doc_id_key:
        chunk_file_doc_id_key = _autodetect_first_key(first_dict, ["doc_id", "docId", "articleId", "paper_id", "paperId", "article_id"]) or ""
        inferred["chunk_file_doc_id_key"] = chunk_file_doc_id_key

    chunk_to_doc: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = get_by_path(it, chunk_file_chunk_id_key)
        if cid is None:
            continue
        cid = str(cid)
        did = get_by_path(it, chunk_file_doc_id_key) if chunk_file_doc_id_key else None
        if did is not None:
            chunk_to_doc[cid] = str(did)

    return {"chunk_to_doc": chunk_to_doc, "inferred_keys": inferred}


def extract_chunk_id_from_group_key(group_key: str) -> str:
    return str(group_key).split("__", 1)[0]


def build_group_maps(
    records: List[Dict[str, Any]],
    field: str,
    *,
    chunk_id_key: str,
    candidate_id_key: str,
    answer_key: str,
    doc_id_key: str,
    chunk_to_doc: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    g_chunk = defaultdict(list)
    g_answer = defaultdict(list)
    g_doc = defaultdict(list)

    for r in records:
        if field not in r:
            continue
        text = str(r[field]).strip()
        if not text:
            continue

        cid = r.get(chunk_id_key, None)
        if cid is None:
            continue
        cid = str(cid)
        g_chunk[cid].append(text)

        cand = r.get(candidate_id_key, None)
        if cand is not None:
            g_answer[f"{cid}__cand__{cand}"].append(text)
        else:
            ans = str(r.get(answer_key, "")).strip()
            key = normalize_text(ans)[:200] if ans else "EMPTY"
            g_answer[f"{cid}__ans__{key}"].append(text)

        did = None
        if doc_id_key and doc_id_key in r and r.get(doc_id_key) is not None:
            did = str(r.get(doc_id_key))
        else:
            did = chunk_to_doc.get(cid)
        if did is not None:
            g_doc[did].append(text)

    return dict(g_chunk), dict(g_answer), dict(g_doc)


def compute_for_groups(
    groups_map: Dict[str, List[str]],
    *,
    encoder: Optional[SimCSEEncoder],
    tokenizer_mode: str,
    bleu_impl: str,
    device: str,
    bertscore_model: str,
    max_group_size: int,
    seed: int,
    diversity_k: int,
    distinct_norm: str,
    distinct_agg: str,
    distinct_lang: str,
    distinct_token_unit: str,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    groups = []
    for sents in groups_map.values():
        sents = [s.strip() for s in sents if s and s.strip()]
        if len(sents) < 2:
            continue
        if max_group_size > 0 and len(sents) > max_group_size:
            idx = rng.choice(len(sents), size=max_group_size, replace=False)
            sents = [sents[i] for i in idx]
        groups.append(sents)

    all_sentences = [x for g in groups for x in g]
    out: Dict[str, Any] = {
        "num_groups_ge2": int(len(groups)),
        "num_sentences_total": int(len(all_sentences)),
        "diversity_k": int(diversity_k),
        "max_group_size": int(max_group_size),
    }

    sample_joined = " ".join(all_sentences[:100])
    if distinct_lang == "auto":
        lang = "en" if _is_likely_en(sample_joined) else "zh"
    else:
        lang = distinct_lang
    if distinct_token_unit == "auto":
        token_unit = "word" if lang == "en" else "char"
    else:
        token_unit = distinct_token_unit

    out["distinct_norm"] = str(distinct_norm)
    out["distinct_agg"] = str(distinct_agg)
    out["distinct_lang"] = str(lang)
    out["distinct_token_unit"] = str(token_unit)

    if distinct_agg == "macro":
        out["distinct_1"] = distinct_n_macro(groups, 1, lang=lang, token_unit=token_unit, norm=distinct_norm)
        out["distinct_2"] = distinct_n_macro(groups, 2, lang=lang, token_unit=token_unit, norm=distinct_norm)
    else:
        out["distinct_1"] = distinct_n_micro(all_sentences, 1, lang=lang, token_unit=token_unit, norm=distinct_norm)
        out["distinct_2"] = distinct_n_micro(all_sentences, 2, lang=lang, token_unit=token_unit, norm=distinct_norm)

    try:
        if bleu_impl == "coco":
            out["self_BLEU_4"] = self_bleu_coco(groups, 4)
        else:
            out["self_BLEU_4"] = self_bleu_sacrebleu(groups, 4)
    except Exception as e:
        out["self_BLEU_4"] = None
        out["self_BLEU_error"] = str(e)

    if encoder is not None and groups:
        cos_scores = []
        vs_scores = []
        divk_scores = []
        divk_sizes = []
        for g in groups:
            emb = encoder.encode(g)
            cos_scores.append(avg_pairwise_cosine(emb))
            try:
                vs_scores.append(vendi_score_embed(emb, q=1.0))
            except Exception:
                pass
            try:
                divk, sel_sz = diversity_at_k_farthest_first(emb, diversity_k, seed=seed)
                if not math.isnan(divk):
                    divk_scores.append(divk)
                    divk_sizes.append(sel_sz)
            except Exception:
                pass

        out["self_cosSim"] = float(np.mean(cos_scores)) if cos_scores else float("nan")
        out["VendiScore"] = float(np.mean(vs_scores)) if vs_scores else float("nan")
        out["Diversity_at_k"] = float(np.mean(divk_scores)) if divk_scores else float("nan")
        out["Diversity_at_k_selected_size_avg"] = float(np.mean(divk_sizes)) if divk_sizes else float("nan")
    else:
        out["self_cosSim"] = float("nan")
        out["VendiScore"] = float("nan")
        out["Diversity_at_k"] = float("nan")
        out["Diversity_at_k_selected_size_avg"] = float("nan")

    try:
        f1, model_used = self_bertscore(
            groups,
            device=device,
            bertscore_model=bertscore_model,
            lang="en",
        )
        out["self_BERTScore_F1"] = f1
        out["self_BERTScore_model_type"] = model_used
    except Exception as e:
        out["self_BERTScore_F1"] = None
        out["self_BERTScore_model_type"] = None
        out["self_BERTScore_error"] = str(e)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--qa_jsonl", required=True)
    ap.add_argument("--field", default="question", choices=["question", "answer", "qa"])
    ap.add_argument("--out_json", default="")

    ap.add_argument("--chunk_id_key", default="chunk_id")
    ap.add_argument("--doc_id_key", default="", help="Doc id key in QA JSONL (e.g., articleId). If empty, try derive from chunks file.")
    ap.add_argument("--candidate_id_key", default="candidate_id")
    ap.add_argument("--question_key", default="question")
    ap.add_argument("--answer_key", default="answer")

    ap.add_argument("--chunks_file", default="", help="Path to chunks JSON/JSONL with chunk metadata.")
    ap.add_argument("--chunk_file_chunk_id_key", default="id", help="Chunk file key for chunk_id.")
    ap.add_argument("--chunk_file_doc_id_key", default="", help="Chunk file key for doc_id (auto-detect if empty).")

    ap.add_argument("--simcse_model", default="", help="Path to the embedding model for SimCSE-style encoding.")
    ap.add_argument("--bertscore_model", default="", help="Path to the encoder model for BERTScore.")
    ap.add_argument("--device", default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)

    ap.add_argument("--tokenizer", default="nltk", choices=["whitespace", "regex", "nltk"])
    ap.add_argument("--distinct_norm", default="content", choices=["raw", "strip_template", "content"])
    ap.add_argument("--distinct_agg", default="macro", choices=["micro", "macro"])
    ap.add_argument("--distinct_lang", default="auto", choices=["auto", "en", "zh"])
    ap.add_argument("--distinct_token_unit", default="auto", choices=["auto", "word", "char"])
    ap.add_argument("--bleu_impl", default="sacrebleu", choices=["sacrebleu", "coco"])
    ap.add_argument("--max_group_size", type=int, default=0, help="cap each group for speed; 0 means no cap")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--diversity_k", type=int, default=10, help="Budget k for Diversity@k (embed).")

    args = ap.parse_args()

    records = load_jsonl(args.qa_jsonl)

    if args.field == "question":
        field_key = args.question_key
    elif args.field == "answer":
        field_key = args.answer_key
    else:
        field_key = "__qa__"

    chunk_res = load_chunk_resources(
        args.chunks_file,
        chunk_file_chunk_id_key=args.chunk_file_chunk_id_key,
        chunk_file_doc_id_key=args.chunk_file_doc_id_key,
    )
    chunk_to_doc = chunk_res["chunk_to_doc"]
    inferred_keys = chunk_res["inferred_keys"]

    if field_key == "__qa__":
        qk, ak = args.question_key, args.answer_key
        for r in records:
            q = str(r.get(qk, "")).strip()
            a = str(r.get(ak, "")).strip()
            r[field_key] = (q + " [SEP] " + a).strip()

    g_chunk, g_answer, g_doc = build_group_maps(
        records,
        field=field_key,
        chunk_id_key=args.chunk_id_key,
        candidate_id_key=args.candidate_id_key,
        answer_key=args.answer_key,
        doc_id_key=args.doc_id_key,
        chunk_to_doc=chunk_to_doc,
    )

    encoder = None
    if torch is not None:
        encoder = SimCSEEncoder(
            model_name=args.simcse_model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    report: Dict[str, Any] = {
        "input": {
            "qa_jsonl": args.qa_jsonl,
            "chunks_file": args.chunks_file,
            "field": args.field,
            "num_records": int(len(records)),
            "chunk_id_key": args.chunk_id_key,
            "doc_id_key": args.doc_id_key,
            "candidate_id_key": args.candidate_id_key,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
            "chunk_file_chunk_id_key": args.chunk_file_chunk_id_key,
            "chunk_file_doc_id_key": args.chunk_file_doc_id_key,
            "inferred_keys_from_chunks_file": inferred_keys,
            "simcse_model": args.simcse_model,
            "bertscore_model": args.bertscore_model,
            "device": args.device,
            "tokenizer": args.tokenizer,
            "distinct_norm": args.distinct_norm,
            "distinct_agg": args.distinct_agg,
            "distinct_lang": args.distinct_lang,
            "distinct_token_unit": args.distinct_token_unit,
            "bleu_impl": args.bleu_impl,
            "max_group_size": int(args.max_group_size),
            "seed": int(args.seed),
            "diversity_k": int(args.diversity_k),
        },
        "doc_level": compute_for_groups(
            g_doc,
            encoder=encoder,
            tokenizer_mode=args.tokenizer,
            bleu_impl=args.bleu_impl,
            device=args.device,
            bertscore_model=args.bertscore_model,
            max_group_size=args.max_group_size,
            seed=args.seed,
            diversity_k=args.diversity_k,
            distinct_norm=args.distinct_norm,
            distinct_agg=args.distinct_agg,
            distinct_lang=args.distinct_lang,
            distinct_token_unit=args.distinct_token_unit,
        ),
    }

    out_str = json.dumps(report, ensure_ascii=False, indent=2)
    print(out_str)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(out_str + "\n")


if __name__ == "__main__":
    main()
