import os
import re
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

import jieba.posseg as pseg
from nltk import word_tokenize, pos_tag


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = _get_device()
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

_EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
_EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "256"))
_USE_FP16 = os.getenv("EMBEDDING_FP16", "1") == "1"

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        if not _EMBEDDING_MODEL_PATH:
            raise RuntimeError("EMBEDDING_MODEL_PATH is not set.")
        _MODEL = SentenceTransformer(_EMBEDDING_MODEL_PATH, device=DEVICE)
        _MODEL.max_seq_length = 256
    return _MODEL


def deduplicate_keywords(keywords: List[str]) -> List[str]:
    seen, out = set(), []
    for k in keywords:
        k2 = (k or "").strip()
        if not k2:
            continue
        if k2 not in seen:
            seen.add(k2)
            out.append(k2)
    return out

def _is_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in (s or ""))

def extract_core_keywords_chinese(text: str) -> List[str]:
    """
    Chinese: use jieba.posseg and keep {n, vn, nt, nz}.
    """
    keep_pos = {"n", "vn", "nt", "nz"}
    kws = [w.word for w in pseg.cut(text or "") if w.flag in keep_pos]
    return deduplicate_keywords(kws)

def extract_core_keywords_english(text: str) -> List[str]:
    """
    English: token + POS only.
    Keep noun-like POS tags and filter long all-caps tokens.
    """
    words = word_tokenize(text or "")
    tags = pos_tag(words)
    keep_pos = {"NN", "NNS", "NNP", "NNPS"}
    kws = []
    for w, t in tags:
        if t in keep_pos:
            if len(w) >= 2 and not (w.isupper() and len(w) >= 3):
                kws.append(w)
    return deduplicate_keywords(kws)

def extract_core_keywords(text: str) -> List[str]:
    """
    Auto route for Chinese vs English.
    """
    return extract_core_keywords_chinese(text) if _is_chinese(text) else extract_core_keywords_english(text)

@torch.no_grad()
def _encode(texts: List[str], normalize: bool = True) -> torch.Tensor:
    if not texts:
        d = _get_model().get_sentence_embedding_dimension()
        return torch.empty(0, d, device=DEVICE)
    if DEVICE == "cuda" and _USE_FP16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return _get_model().encode(
                texts,
                batch_size=_EMBEDDING_BATCH,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
    else:
        return _get_model().encode(
            texts,
            batch_size=_EMBEDDING_BATCH,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

@lru_cache(maxsize=4096)
def _encode_cached(line: str) -> torch.Tensor:
    return _encode([line])

@torch.no_grad()
def keyword_cosine_similarity(text1: str, text2: str) -> float:
    """
    Bi-directional max-match cosine similarity over keyword sets.
    """
    kw1, kw2 = extract_core_keywords(text1 or ""), extract_core_keywords(text2 or "")
    if not kw1 or not kw2:
        return 0.0

    e1 = _encode(kw1)  # (n1, d)
    e2 = _encode(kw2)  # (n2, d)

    sim = util.cos_sim(e1, e2).cpu().numpy()  # (n1, n2)
    max1 = np.max(sim, axis=1)
    max2 = np.max(sim, axis=0)
    return float((np.mean(max1) + np.mean(max2)) / 2.0)

@torch.no_grad()
def semantic_similarity(text1: str, text2: str) -> float:
    """
    Sentence-level semantic similarity.
    """
    e1 = _encode_cached(text1).squeeze(0)
    e2 = _encode_cached(text2).squeeze(0)
    if e1.numel() == 0 or e2.numel() == 0:
        return 0.0
    return float(util.cos_sim(e1, e2).item())

def compute_similarity(text1: str, text2: str, weight_keyword: float = 0.5, weight_semantic: float = 0.5) -> float:
    """
    Weighted sum of keyword similarity and semantic similarity.
    """
    sk = keyword_cosine_similarity(text1, text2)
    ss = semantic_similarity(text1, text2)
    return float(weight_keyword * sk + weight_semantic * ss)

@torch.no_grad()
def extract_representative_keywords(text: str, candidate_answer: str, top_k: int = 10, thresh: float = 0.40) -> List[str]:
    """
    Select keywords from the text that are close to the candidate answer.
    """
    kw_text = extract_core_keywords(text or "")
    if not kw_text:
        return []

    kw_emb = _encode(kw_text)                # (n, d)
    ans_emb = _encode_cached(candidate_answer)  # (1, d)

    sim = util.cos_sim(kw_emb, ans_emb).squeeze(1)  # (n,)
    kept = [kw_text[i] for i in range(len(kw_text)) if float(sim[i].item()) > thresh]
    kept = deduplicate_keywords(kept)
    return kept[:top_k]

@torch.no_grad()
def compute_max_sentence_sim(sent: str, passage: str) -> float:
    sents = re.split(r"[.!?\n\r]+", passage or "")
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return 0.0
    emb_sent = _encode_cached(sent)     # (1, d)
    emb_sents = _encode(sents)          # (m, d)
    sim = util.cos_sim(emb_sents, emb_sent).squeeze(1)  # (m,)
    return float(sim.max().item())
