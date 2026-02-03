# vrqa/router/features.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


_EMBEDDING_DEVICE: str = "cuda"
_SEM_ENCODER = None

# simple caches
_DOC_SEM_CACHE: Dict[str, List[float]] = {}


def set_embedding_device(device: str) -> None:
    """Set embedding device for SentenceTransformer ("cuda" or "cpu")."""
    global _EMBEDDING_DEVICE
    name = (device or "").strip().lower()
    if name not in {"cuda", "cpu"}:
        name = "cuda"
    if name == "cuda":
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                name = "cpu"
        except Exception:
            name = "cpu"
    _EMBEDDING_DEVICE = name


def _ensure_numpy() -> bool:
    global np
    if np is not None:
        return True
    try:
        import numpy as _np  # type: ignore
        np = _np  # type: ignore
        return True
    except Exception:
        return False


def _get_semantic_encoder():
    """
    Lazy-load SentenceTransformer from env EMBEDDING_MODEL_PATH.
    """
    global _SEM_ENCODER
    if _SEM_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model_path = (os.environ.get("EMBEDDING_MODEL_PATH") or "").strip()
        if not model_path:
            raise RuntimeError("EMBEDDING_MODEL_PATH is not set.")
        _SEM_ENCODER = SentenceTransformer(model_path, device=_EMBEDDING_DEVICE)
    return _SEM_ENCODER


def _seed_from_kind(kind: str) -> int:
    # fixed + deterministic seeds per kind
    base = 1337
    h = 0
    for ch in (kind or ""):
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return base ^ h


def _project_embedding(vec: "np.ndarray", kind: str, out_dim: int = 16) -> List[float]:
    """
    Fixed random projection to out_dim (deterministic).
    """
    if not _ensure_numpy() or np is None:
        return [0.0] * out_dim
    v = np.asarray(vec, dtype=float).reshape(-1)
    dim_full = int(v.shape[0])
    if dim_full <= 0:
        return [0.0] * out_dim

    rng = np.random.RandomState(_seed_from_kind(kind))
    R = rng.normal(size=(dim_full, out_dim)) / math.sqrt(dim_full)
    proj = v @ R
    return [float(x) for x in proj.reshape(-1).tolist()]


def _encode_text(text: str) -> Optional["np.ndarray"]:
    if not text:
        return None
    enc = _get_semantic_encoder()
    vec = enc.encode(text, normalize_embeddings=True)
    if not _ensure_numpy() or np is None:
        return None
    return np.asarray(vec, dtype=float).reshape(-1)


def get_doc_semantic(article_meta: Dict[str, Any], out_dim: int = 16) -> List[float]:
    """
    Doc-level semantic vector from title + abstract + keywords (cached).
    """
    meta = article_meta or {}
    title = str(meta.get("title") or "").strip()
    abstract = str(meta.get("abstract") or "").strip()
    kws = meta.get("keywords") or []
    kw_text = ", ".join(str(k) for k in kws if k)

    doc_text = " ".join([title, abstract, kw_text]).strip()
    if not doc_text:
        return [0.0] * out_dim

    if doc_text in _DOC_SEM_CACHE:
        return _DOC_SEM_CACHE[doc_text]

    vec = _encode_text(doc_text)
    if vec is None:
        sem = [0.0] * out_dim
    else:
        sem = _project_embedding(vec, kind="doc", out_dim=out_dim)

    _DOC_SEM_CACHE[doc_text] = sem
    return sem


def get_chunk_semantic(chunk_summary: str, chunk_keywords: List[str], out_dim: int = 16) -> List[float]:
    """
    Chunk-level semantic vector from summary + keywords.
    """
    sum_text = str(chunk_summary or "").strip()
    kw_text = ", ".join(str(k) for k in (chunk_keywords or []) if k)
    chunk_text = " ".join([sum_text, kw_text]).strip()
    if not chunk_text:
        return [0.0] * out_dim

    vec = _encode_text(chunk_text)
    if vec is None:
        return [0.0] * out_dim
    return _project_embedding(vec, kind="chunk", out_dim=out_dim)


def get_candidate_semantic(candidate_answer: str, out_dim: int = 16) -> List[float]:
    """
    Candidate answer semantic vector.
    """
    cand_text = str(candidate_answer or "").strip()
    if not cand_text:
        return [0.0] * out_dim
    vec = _encode_text(cand_text)
    if vec is None:
        return [0.0] * out_dim
    return _project_embedding(vec, kind="cand", out_dim=out_dim)


def build_chunk_context_features(
    content: str,
    chunk_summary: str,
    chunk_keywords: List[str],
    article_meta: Dict[str, Any],
    candidate_answer: str,
    *,
    out_dim: int = 16,
) -> Tuple[List[float], int]:
    """
    Build x_ctx for LinUCB routing.

    Current default: [chunk_sem(16) + cand_sem(16)] => dim=32.
    (你后面如果要加 doc_sem，也可以直接拼进去。)
    """
    chunk_sem = get_chunk_semantic(chunk_summary, chunk_keywords or [], out_dim=out_dim)
    cand_sem = get_candidate_semantic(candidate_answer, out_dim=out_dim)
    x = chunk_sem + cand_sem
    return x, len(x)


def build_view_semantic_embeddings(view_desc: Dict[str, str], out_dim: int = 16) -> Dict[str, List[float]]:
    """
    Build semantic embeddings for view descriptions.
    Returns: {view: embedding_list}
    """
    if not view_desc:
        return {}
    try:
        enc = _get_semantic_encoder()
    except Exception:
        return {}
    out: Dict[str, List[float]] = {}
    for v, desc in view_desc.items():
        text = f"{v}: {desc}".strip() if desc else str(v)
        try:
            vec = enc.encode(text, normalize_embeddings=True)
            if not _ensure_numpy() or np is None:
                continue
            arr = np.asarray(vec, dtype=float)
            out[v] = _project_embedding(arr, kind="view", out_dim=out_dim)
        except Exception:
            continue
    return out
