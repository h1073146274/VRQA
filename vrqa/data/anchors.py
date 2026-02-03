from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclass
class LlmConfig:
    api_key: str
    api_base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_sec: int = 120


def load_llm_config_from_env() -> Optional[LlmConfig]:
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    api_base_url = (os.getenv("LLM_API_BASE_URL") or "").strip()
    model = (os.getenv("LLM_MODEL") or "").strip()
    if not api_key or not api_base_url:
        return None
    if not model:
        model = "default"
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
    timeout_sec = int(os.getenv("LLM_TIMEOUT_SEC", "120"))
    return LlmConfig(
        api_key=api_key,
        api_base_url=api_base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def _resolve_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _extract_text_from_response(resp_json: Any) -> str:
    if isinstance(resp_json, str):
        return resp_json.strip()
    if not isinstance(resp_json, dict):
        return ""
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"].strip()
            if isinstance(first.get("text"), str):
                return first["text"].strip()
            if isinstance(first.get("content"), str):
                return first["content"].strip()
    if isinstance(resp_json.get("content"), str):
        return resp_json["content"].strip()
    return ""


def _parse_summary_keywords(raw: str) -> Tuple[str, List[str]]:
    if not raw:
        return "", []

    raw_stripped = raw.strip()
    if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
        try:
            obj = json.loads(raw_stripped)
            summary = str(obj.get("summary") or "").strip()
            keywords = obj.get("keywords") or []
            if isinstance(keywords, str):
                keywords = re.split(r"[,;\n\r\t]+", keywords)
            keywords = [k.strip() for k in keywords if k and k.strip()]
            return summary, keywords
        except Exception:
            pass

    sum_match = re.search(r"(?i)summary:?\s*(.+?)(?:\n[A-Z][a-zA-Z ]*:|$)", raw_stripped, re.DOTALL)
    kw_match = re.search(r"(?i)keywords?:?\s*(.+)", raw_stripped)
    summary = sum_match.group(1).strip() if sum_match else ""
    keywords_str = kw_match.group(1).strip() if kw_match else ""
    keywords_str = re.sub(r"(?:^|\n)\s*(?:\d+\.\s*|[-*]\s*)", "\n", keywords_str)
    keywords = [k.strip(" -\t") for k in re.split(r"[,;\n\r\t]+", keywords_str) if k.strip()]
    return summary, keywords


def _simple_keywords(text: str, top_k: int = 6) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if not words:
        return []
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "over", "under",
        "are", "was", "were", "be", "been", "being", "is", "of", "to", "in", "on",
        "as", "by", "an", "a", "or", "at", "it", "its", "their", "they", "we",
    }
    counts: Dict[str, int] = {}
    for w in words:
        if w in stop or len(w) <= 2:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:top_k]]


def _simple_summary(text: str, max_chars: int = 240) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = sentences[0].strip() if sentences else text
    return summary[:max_chars].strip()


def _build_doc_prompt(title: str, text: str) -> str:
    return (
        "You are an expert at extracting a concise summary and keywords.\n"
        "Analyze the following document and produce a short summary and keywords.\n\n"
        "Requirements:\n"
        "1) Summary in 1 short paragraph.\n"
        "2) 3-6 keywords, comma-separated.\n"
        "3) Output strictly in JSON: {\"summary\": \"...\", \"keywords\": [\"k1\", \"k2\"]}\n\n"
        f"[Title]\n{title}\n\n[Document]\n{text}"
    )


def _build_chunk_prompt(text: str) -> str:
    return (
        "You are an expert at extracting a short summary and keywords for a paragraph.\n"
        "Requirements:\n"
        "1) Summary in 1 sentence.\n"
        "2) 2-4 keywords, comma-separated.\n"
        "3) Output strictly in JSON: {\"summary\": \"...\", \"keywords\": [\"k1\", \"k2\"]}\n\n"
        f"[Paragraph]\n{text}"
    )


class AnchorExtractor:
    def __init__(self, backend: str = "llm", llm: Optional[LlmConfig] = None, sleep_sec: float = 0.2) -> None:
        self.backend = (backend or "llm").strip().lower()
        self.llm = llm
        self.sleep_sec = float(sleep_sec)

    def _call_llm(self, prompt: str) -> str:
        if not self.llm:
            return ""
        url = _resolve_chat_url(self.llm.api_base_url)
        headers = {
            "Authorization": f"Bearer {self.llm.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.llm.timeout_sec)
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = {"content": resp.text or ""}
        return _extract_text_from_response(resp_json)

    def _extract_with_llm(self, prompt: str) -> Tuple[str, List[str]]:
        raw = self._call_llm(prompt)
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        return _parse_summary_keywords(raw)

    def _extract_with_heuristic(self, text: str) -> Tuple[str, List[str]]:
        summary = _simple_summary(text)
        keywords = _simple_keywords(text)
        return summary, keywords

    def extract_doc_anchor(
        self,
        title: str,
        text: str,
        lang: str,
        given_abstract: str = "",
        given_keywords: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        abstract = (given_abstract or "").strip()
        keywords = [k for k in (given_keywords or []) if str(k).strip()]
        if abstract and keywords:
            return abstract, keywords

        if self.backend == "llm" and self.llm:
            summary, kws = self._extract_with_llm(_build_doc_prompt(title, text))
        else:
            summary, kws = self._extract_with_heuristic(text)

        if not abstract:
            abstract = summary
        if not keywords:
            keywords = kws
        return abstract, keywords

    def extract_chunk_anchor(self, chunk_text: str, lang: str) -> Tuple[str, List[str]]:
        if self.backend == "llm" and self.llm:
            return self._extract_with_llm(_build_chunk_prompt(chunk_text))
        return self._extract_with_heuristic(chunk_text)
