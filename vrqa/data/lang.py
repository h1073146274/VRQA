# vrqa/data/lang.py
import re

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

def detect_language(text: str) -> str:
    """
    Heuristic lang detect: 'zh' if CJK dominates else 'en'.
    """
    if not text:
        return "en"
    cjk = len(_CJK_RE.findall(text))
    latin = len(_ASCII_LETTER_RE.findall(text))
    total = max(len(text), 1)
    if cjk / total >= 0.05 or cjk >= 16:
        return "zh"
    if latin / total >= 0.05 or latin >= 20:
        return "en"
    return "en"
