# vrqa/data/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

Json = Union[Dict[str, Any], List[Any]]

def load_json_or_jsonl(path: str) -> Json:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(text)

def write_json(path: str, obj: Any, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
