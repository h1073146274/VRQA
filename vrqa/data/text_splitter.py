# vrqa/data/text_splitter.py
from __future__ import annotations
import re
from typing import List

_SENT_END_EN = re.compile(r"(?<=[.!?])\s+")
_SENT_END_ZH = re.compile(r"(?<=[.!?;])")

def _tail_with_word_boundary(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    cut = s[-n:]
    # try avoid cutting in middle of a word for EN
    m = re.search(r"\b\w+\b", cut)
    return cut if not m else cut[m.start():]

def split_text_into_chunks(
    text: str,
    min_chars: int = 800,
    max_chars: int = 2200,
    overlap_chars: int = 150,
    lang: str = "en",
) -> List[str]:
    """
    Sentence-ish packing with soft constraints:
      1) split into sentences
      2) pack to max_chars
      3) merge too-short blocks
      4) inject overlap between adjacent blocks
    """
    text = (text or "").strip()
    if not text:
        return []

    if lang == "zh":
        sents = [s.strip() for s in _SENT_END_ZH.split(text) if s.strip()]
    else:
        sents = [s.strip() for s in _SENT_END_EN.split(text) if s.strip()]

    chunks: List[str] = []
    buf = ""
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                chunks.append(buf.strip())
            # very long single sentence fallback
            if len(s) > max_chars:
                # hard cut
                start = 0
                while start < len(s):
                    chunks.append(s[start:start + max_chars].strip())
                    start += max_chars
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf.strip())

    # merge short blocks
    if min_chars and min_chars < max_chars and len(chunks) > 1:
        merged: List[str] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            while (len(cur) < min_chars) and (i + 1 < len(chunks)) and (len(cur) + 2 + len(chunks[i + 1]) <= max_chars):
                i += 1
                cur = cur + "\n\n" + chunks[i]
            merged.append(cur.strip())
            i += 1
        chunks = merged

    # overlap injection
    if overlap_chars > 0 and len(chunks) > 1:
        with_overlap: List[str] = [chunks[0]]
        for j in range(1, len(chunks)):
            prev_tail = _tail_with_word_boundary(chunks[j - 1], overlap_chars)
            cand = (prev_tail + ("\n" if prev_tail and not prev_tail.endswith("\n") else "") + chunks[j]).strip()
            with_overlap.append(cand if len(cand) <= max_chars else chunks[j])
        chunks = with_overlap

    return [c for c in chunks if c]
