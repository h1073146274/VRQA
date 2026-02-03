# vrqa/data/prepare_chunks.py
from __future__ import annotations
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .anchors import AnchorExtractor, load_llm_config_from_env
from .io import load_json_or_jsonl, write_jsonl
from .lang import detect_language
from .text_splitter import split_text_into_chunks

def _as_list_keywords(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        parts = [p.strip() for p in re.split(r"[,;\n\r\t]+", x) if p.strip()]
        return parts
    s = str(x).strip()
    return [s] if s else []

def _get_text(article: Dict[str, Any]) -> str:
    for key in ("text", "content", "body"):
        v = article.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # sometimes body is list of paragraphs
    v = article.get("paragraphs")
    if isinstance(v, list):
        return "\n\n".join(str(x) for x in v if x is not None).strip()
    return ""

def _get_article_id(article: Dict[str, Any]) -> str:
    for k in ("articleId", "article_id", "doc_id", "paper_id", "id"):
        v = article.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return str(uuid.uuid4())

def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

@dataclass
class PrepareConfig:
    min_chars: int = 800
    max_chars: int = 2200
    overlap_chars: int = 150
    backend: str = "llm"          # "llm" or "heuristic"
    sleep_sec: float = 0.2
    max_docs: int = -1            # for debugging

def iter_articles(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        # allow {"data":[...]} or dict of id->record
        if isinstance(obj.get("data"), list):
            for x in obj["data"]:
                if isinstance(x, dict):
                    yield x
        else:
            for x in obj.values():
                if isinstance(x, dict):
                    yield x

def prepare_chunks(
    input_path: str,
    output_path: str,
    cfg: PrepareConfig,
    cache_dir: Optional[str] = None,
) -> None:
    raw = load_json_or_jsonl(input_path)
    articles = list(iter_articles(raw))
    if cfg.max_docs and cfg.max_docs > 0:
        articles = articles[: cfg.max_docs]

    llm_cfg = load_llm_config_from_env()
    if cfg.backend == "llm" and llm_cfg is None:
        raise RuntimeError("backend=llm but LLM_API_KEY/LLM_API_BASE_URL not set (see .env.example)")

    extractor = AnchorExtractor(backend=cfg.backend, llm=llm_cfg, sleep_sec=cfg.sleep_sec)

    cache_p = Path(cache_dir) if cache_dir else None
    doc_cache: Dict[str, Any] = {}
    chunk_cache: Dict[str, Any] = {}
    if cache_p:
        cache_p.mkdir(parents=True, exist_ok=True)
        doc_cache_path = cache_p / "doc_anchor_cache.json"
        chunk_cache_path = cache_p / "chunk_anchor_cache.json"
        if doc_cache_path.exists():
            doc_cache = json.loads(doc_cache_path.read_text(encoding="utf-8"))
        if chunk_cache_path.exists():
            chunk_cache = json.loads(chunk_cache_path.read_text(encoding="utf-8"))

    out_rows: List[Dict[str, Any]] = []
    for idx, art in enumerate(articles, 1):
        title = (art.get("title") or art.get("paper_title") or "").strip()
        text = _get_text(art)
        if not text:
            continue

        article_id = _get_article_id(art)
        lang = (art.get("language") or "").lower()
        lang = lang if lang in ("en", "zh") else detect_language(f"{title}\n{text}")

        given_abs = (art.get("abstract") or "").strip()
        given_kw = art.get("keywords")
        if isinstance(given_kw, str):
            given_kw = [k.strip() for k in re.split(r"[,;\n\r\t]+", given_kw) if k.strip()]
        if not isinstance(given_kw, list):
            given_kw = []

        # --- doc anchor (cached) ---
        doc_key = f"{article_id}:{_hash_text((title+'|'+given_abs+'|'+str(given_kw))[:2000])}"
        if doc_key in doc_cache:
            meta_abs = doc_cache[doc_key]["abstract"]
            meta_kw = doc_cache[doc_key]["keywords"]
        else:
            meta_abs, meta_kw = extractor.extract_doc_anchor(
                title=title,
                text=(f"{title}\n\n{text}" if title else text),
                lang=lang,
                given_abstract=given_abs,
                given_keywords=given_kw,
            )
            doc_cache[doc_key] = {"abstract": meta_abs, "keywords": meta_kw}

        # --- split into chunks ---
        chunks = split_text_into_chunks(
            text,
            min_chars=cfg.min_chars,
            max_chars=cfg.max_chars,
            overlap_chars=cfg.overlap_chars,
            lang=lang,
        )

        for ci, chunk_text in enumerate(chunks, 1):
            # --- chunk anchor (cached) ---
            ch_key = f"{article_id}:{ci}:{_hash_text(chunk_text)}"
            if ch_key in chunk_cache:
                ch_sum = chunk_cache[ch_key]["summary"]
                ch_kw = chunk_cache[ch_key]["keywords"]
            else:
                ch_sum, ch_kw = extractor.extract_chunk_anchor(chunk_text, lang=lang)
                chunk_cache[ch_key] = {"summary": ch_sum, "keywords": ch_kw}

            out_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "articleId": article_id,
                    "title": title,
                    "content": chunk_text,
                    "summary": ch_sum or "Summary unavailable",
                    "paragraph_keywords": ch_kw or [],
                    "length": len(chunk_text),
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "meta": {
                        "abstract": meta_abs or "Summary unavailable",
                        "keywords": meta_kw or [],
                        "domain": (art.get("domain") or art.get("category") or ""),
                    },
                    "language": lang,
                }
            )

        print(f"[{idx}/{len(articles)}] {title[:60] if title else 'Untitled'} -> {len(chunks)} chunks")

    # persist caches
    if cache_p:
        (cache_p / "doc_anchor_cache.json").write_text(json.dumps(doc_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        (cache_p / "chunk_anchor_cache.json").write_text(json.dumps(chunk_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # output jsonl
    write_jsonl(output_path, out_rows)
    print(f"[OK] wrote {len(out_rows)} chunks -> {output_path}")
