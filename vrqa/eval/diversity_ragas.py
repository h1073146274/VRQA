#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, re, importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
def _import_object(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _parse_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse kwargs JSON: {exc}") from exc


def _build_component(class_path: str, kwargs_json: Optional[str]):
    cls = _import_object(class_path)
    kwargs = _parse_kwargs(kwargs_json)
    return cls(**kwargs)

_FaithfulnessClass = None
_AnswerRelevancyClass = None
_faithfulness_metric = None
_answer_relevancy_metric = None
try:
    from ragas.metrics import Faithfulness as _FaithfulnessClass
except Exception:
    pass
try:
    from ragas.metrics import AnswerRelevancy as _AnswerRelevancyClass
except Exception:
    pass
try:
    from ragas.metrics import faithfulness as _faithfulness_metric
except Exception:
    pass
try:
    from ragas.metrics import answer_relevancy as _answer_relevancy_metric
except Exception:
    pass

def _resolve_metric(name: str):
    name = name.lower()
    if name == "faithfulness":
        if _FaithfulnessClass: return _FaithfulnessClass()
        if _faithfulness_metric: return _faithfulness_metric
    if name == "answer_relevancy":
        if _AnswerRelevancyClass: return _AnswerRelevancyClass()
        if _answer_relevancy_metric: return _answer_relevancy_metric
    return None

def load_any(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        text = path.read_text(encoding="utf-8")
        try:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        except json.JSONDecodeError:
            # fallback to treating the file as a regular JSON array/document
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL file {path}: {exc}") from exc
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

def _stringify_chunk_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(x) for x in value if x is not None)
    return str(value)

def normalize_qa_rows(loaded) -> List[Dict[str, Any]]:
    if isinstance(loaded, list):
        if loaded and isinstance(loaded[0], dict): return loaded
        raise ValueError("QA list items must be dicts.")
    if isinstance(loaded, dict):
        if isinstance(loaded.get("chunk_results"), list):
            rows=[]
            for ch in loaded["chunk_results"]:
                for r in ch.get("accepted_results") or []:
                    rr = dict(r)
                    if "id" not in rr:
                        rid = rr.get("candidate_id")
                        rr["id"] = f"{rr.get('chunk_id','NA')}-{rid if rid is not None else 'NA'}"
                    rows.append(rr)
            if not rows:
                raise ValueError("chunk_results has no accepted_results.")
            return rows
        rows=[]
        for v in loaded.values():
            if isinstance(v, list):
                rows.extend([x for x in v if isinstance(x, dict)])
        if rows: return rows
    raise ValueError("Failed to extract QA rows from JSON.")

def load_chunks_map(chunk_path: Optional[str], key_id="id", key_text="content") -> Optional[Dict[str, str]]:
    if not chunk_path: return None
    loaded = load_any(Path(chunk_path))
    if isinstance(loaded, list):
        out={}
        for it in loaded:
            if isinstance(it, dict):
                cid = it.get(key_id) or it.get("chunk_id")
                ctext = it.get(key_text) or it.get("chunk_text") or it.get("content")
                if cid and ctext: out[str(cid)] = _stringify_chunk_text(ctext)
        return out
    if isinstance(loaded, dict):
        out={}
        for k, v in loaded.items():
            text = _stringify_chunk_text(v)
            if text:
                out[str(k)] = text
        return out
    raise ValueError("Chunks file must be a list or dict.")

def _file_contains_any(path: Path, substrings: List[str]) -> bool:
    if not substrings:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return False
    for substring in substrings:
        if substring and substring in text:
            return True
    return False

def auto_discover_chunk_map(
    qa_file: Path,
    rows: List[Dict[str, Any]],
    chunk_id_key: str,
    chunk_text_key: str,
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    When --chunks_file is omitted, try to locate a nearby chunks file that contains one of the chunk_ids.
    """
    if not rows:
        return None, None
    chunk_ids=[]
    for row in rows:
        candidate = row.get(chunk_id_key) or row.get("chunk_id") or row.get("id")
        if candidate:
            chunk_ids.append(str(candidate))
            if len(chunk_ids) >= 20:
                break
    if not chunk_ids:
        return None, None

    search_dirs = [qa_file.parent]
    parent = qa_file.parent.parent
    if parent != qa_file.parent:
        search_dirs.append(parent)
    patterns = ["*chunks*.json", "*chunk_map*.json", "*chunks*.jsonl", "*chunk_map*.jsonl"]

    for root in search_dirs:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for candidate in sorted(root.glob(pattern)):
                if candidate == qa_file or not candidate.is_file():
                    continue
                if not _file_contains_any(candidate, chunk_ids):
                    continue
                try:
                    cand_map = load_chunks_map(str(candidate), key_id=chunk_id_key, key_text=chunk_text_key)
                except Exception:
                    continue
                if cand_map:
                    matching = next((cid for cid in chunk_ids if cid in cand_map), None)
                    if matching:
                        print(f"[AUTO] loaded context from {candidate} (chunk_id={matching})")
                        return str(candidate), cand_map
    return None, None

def resolve_context_from_row(
    row: Dict[str, Any],
    chunk_map: Optional[Dict[str, str]],
    chunk_id_key: str,
    chunk_text_key: str,
) -> Optional[str]:
    """
    Try to pick up a context string from the QA row, optionally falling back to the chunk map.
    """
    text = row.get(chunk_text_key) or row.get("context_text") or row.get("context") or row.get("chunk_text")
    if text:
        return str(text)
    chunk_id = row.get(chunk_id_key) or row.get("chunk_id") or row.get("id")
    if chunk_id is None or chunk_map is None:
        return None
    return chunk_map.get(str(chunk_id))


def resolve_field(row: Dict[str, Any], primary: str, fallbacks: List[str]) -> Optional[str]:
    """
    Fetch a value, trying fallbacks if the primary key is missing.
    """
    value = row.get(primary)
    if value:
        return str(value)
    for key in fallbacks:
        backup = row.get(key)
        if backup:
            return str(backup)
    return None

def to_pct_raw(x: Optional[float]) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return float(max(0.0, min(100.0, 100.0 * v)))


_EN_PUNC = r"""[\s,.;:!?'"`~@#$%^&*_\-+=/\\|<>\[\](){}]"""
EN_PUNCS = re.compile(_EN_PUNC)

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

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
def is_likely_en(text: str) -> bool:
    if not text: return False
    ascii_cnt = sum(1 for ch in text if ord(ch) < 128)
    return (ascii_cnt / max(1, len(text))) > 0.8

def decide_lang_and_unit(sample_texts: List[str], lang_arg: str, unit_arg: str) -> Tuple[str, str]:
    if lang_arg in ("zh", "en"):
        lang = "en"
    else:
        lang = "en" if is_likely_en(" ".join(sample_texts[:100])) else "en"
    if unit_arg in ("char", "word"):
        unit = unit_arg
    else:
        unit = "word"
    return lang, unit

def strip_template_prefix(s: str, lang: str) -> str:
    s = s or ""
    s = _EN_Q_PREFIX.sub("", s)
    s = EN_PUNCS.sub(" ", s)
    return re.sub(r"\s+"," ", s).strip()

def remove_function_words(q: str, lang: str, token_unit: str, *, is_question: bool=True) -> str:
    q = q or ""
    if is_question:
        q = strip_template_prefix(q, lang)
    else:
        q = EN_PUNCS.sub(" ", q)
        q = re.sub(r"\s+"," ", q).strip()
    return re.sub(r"\s+"," ", q).strip()

def conditional_remove(a_text: str, b_text: str, lang: str, token_unit: str, *, a_is_question: bool) -> str:
    a = strip_template_prefix(a_text or "", lang) if a_is_question else (a_text or "")
    b = (b_text or "").strip()
    if not b:
        return a
    a_words = [w for w in re.split(r"\s+", EN_PUNCS.sub(" ", a)) if w]
    b_words = set([w for w in re.split(r"\s+", EN_PUNCS.sub(" ", b)) if w])
    kept = [w for w in a_words if w not in b_words]
    return " ".join(kept)

def token_seq(text: str, *, lang: str, token_unit: str) -> List[str]:
    text = text or ""
    if token_unit=="word":
        text2 = EN_PUNCS.sub(" ", text)
        words = [w for w in re.split(r"\s+", text2) if w]
        return [w for w in words if w.lower() not in _EN_STOP]
    else:
        return [ch for ch in text if not ch.isspace()]

def ngram_list(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if n>=1 else []

def distinct_n_from_texts(texts: List[str], n: int, *, lang: str, token_unit: str) -> float:
    all_ngrams, total = set(), 0
    for t in texts:
        toks = token_seq(t, lang=lang, token_unit=token_unit)
        if len(toks) < n:
            continue
        grams = ngram_list(toks, n)
        total += len(grams)
        all_ngrams.update(grams)
    return (len(all_ngrams)/total) if total else 0.0

def normalize_for_diversity(text: str, other: str, mode: str, *, lang: str, token_unit: str, is_question: bool) -> str:
    if mode=="raw":
        return (text or "").strip()
    if mode=="strip_template":
        return strip_template_prefix(text or "", lang) if is_question else (text or "")
    if mode=="content":
        return remove_function_words(text or "", lang, token_unit, is_question=is_question)
    if mode=="conditional":
        return conditional_remove(text or "", other or "", lang, token_unit, a_is_question=is_question)
    return (text or "").strip()

def compute_distinct_bundle_centeranchor(
    df: pd.DataFrame, *, target: str = "question", n: int = 2, group_by: str = "chunk_id",
    lang: str = "zh", token_unit: str = "char", keep: List[str] = None
) -> Dict[str, float]:
    """
    Four modes: raw / content / conditional / macro (within-group strip_template)
    target: "question" or "answer"
    """
    if keep is None: keep = ["content","macro"]
    tgt_col = "question" if target=="question" else "answer"
    other_col = "answer" if target=="question" else "question"

    texts = df.get(tgt_col, pd.Series([], dtype=str)).astype(str).tolist()
    othrs = df.get(other_col, pd.Series([""]*len(texts), dtype=str)).astype(str).tolist()
    gids  = df.get(group_by, pd.Series([""]*len(texts), dtype=str)).astype(str).tolist()

    is_q = (target=="question")

    bundles: Dict[str, float] = {}
    prefix = f"distinct_{target}-{n}"

    # raw
    vals_raw = [normalize_for_diversity(t, o, "raw", lang=lang, token_unit=token_unit, is_question=is_q) for t,o in zip(texts, othrs)]
    d_raw = distinct_n_from_texts(vals_raw, n=n, lang=lang, token_unit=token_unit)
    bundles[f"{prefix}_raw"] = d_raw

    # content
    vals_content = [normalize_for_diversity(t, o, "content", lang=lang, token_unit=token_unit, is_question=is_q) for t,o in zip(texts, othrs)]
    d_content = distinct_n_from_texts(vals_content, n=n, lang=lang, token_unit=token_unit)
    bundles[f"{prefix}_content"] = d_content

    # conditional
    vals_cond = [normalize_for_diversity(t, o, "conditional", lang=lang, token_unit=token_unit, is_question=is_q) for t,o in zip(texts, othrs)]
    d_cond = distinct_n_from_texts(vals_cond, n=n, lang=lang, token_unit=token_unit)
    bundles[f"{prefix}_conditional"] = d_cond

    bucket: Dict[str, List[str]] = {}
    for t,g in zip(texts, gids):
        bucket.setdefault(g, []).append(normalize_for_diversity(t, "", "strip_template", lang=lang, token_unit=token_unit, is_question=is_q))
    d_macro = 0.0
    if bucket:
        per=[]
        for arr in bucket.values():
            per.append(distinct_n_from_texts(arr, n=n, lang=lang, token_unit=token_unit))
        d_macro = float(np.mean(per)) if per else 0.0
    bundles[f"{prefix}_macro_by_{group_by}"] = d_macro

    out={}
    for k,v in bundles.items():
        ok = (
            ("_raw" in k and "raw" in keep) or
            ("_content" in k and "content" in keep) or
            ("_conditional" in k and "conditional" in keep) or
            ("_macro_by_" in k and "macro" in keep)
        )
        if ok: out[k]=v
    return out

def run_ragas(
    samples: List[SingleTurnSample],
    metrics,
    llm_wrapper,
    emb_wrapper,
    *,
    batch_size=None,
    chunk_size=32,
    max_retries=6,
    backoff_base=3.0,
    backoff_factor=2.0,
    backoff_max=60.0,
    show_progress=True,
):
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kw):  # type: ignore
            return x

    import time, random
    import pandas as pd
    from ragas import evaluate, EvaluationDataset

    def _is_rate_limit_error(err: Exception) -> bool:
        msg = f"{type(err).__name__}: {err}".lower()
        return ("429" in msg) or ("rate limit" in msg) or ("tpm" in msg) or ("request was rejected due to rate limiting" in msg)

    def _is_transient_server_error(err: Exception) -> bool:
        msg = f"{type(err).__name__}: {err}".lower()
        return (
            "503" in msg
            or "50603" in msg
            or "system is too busy" in msg
            or "temporarily" in msg
            or "timeout" in msg
            or "timed out" in msg
            or "read timeout" in msg
        )

    def _sleep_with_jitter(base_s: float, factor: float, attempt: int, max_s: float, jitter: float = 0.25):
        wait = min(max_s, base_s * (factor ** attempt))
        wait = max(0.0, wait * (1.0 + random.uniform(-jitter, jitter)))
        time.sleep(wait)

    n = len(samples)
    if n == 0:
        return pd.DataFrame()

    all_chunks = []
    start = 0
    block_idx = 0

    pbar_total = tqdm(total=n, desc="RAGAS evaluating", unit="sample", disable=not show_progress)

    while start < n:
        end = min(start + chunk_size, n)
        block_idx += 1

        print(f"[BLOCK] #{block_idx}  samples[{start}:{end})  (size={end - start})")
        pbar_block = tqdm(total=(end - start), desc=f"Block #{block_idx}", unit="sample", leave=False, disable=not show_progress)

        dataset_chunk = EvaluationDataset(samples=samples[start:end])

        attempt = 0
        while True:
            try:
                res = evaluate(
                    dataset=dataset_chunk,
                    metrics=metrics,
                    show_progress=False,
                    batch_size=batch_size,
                    llm=llm_wrapper,
                    embeddings=emb_wrapper,
                )
                df_chunk = res.to_pandas() if hasattr(res, "to_pandas") else pd.DataFrame()
                all_chunks.append(df_chunk)

                pbar_block.update(end - start)
                pbar_block.close()
                pbar_total.update(end - start)

                print(f"[DONE]     #{block_idx}  samples[{start}:{end})")
                break

            except Exception as e:
                attempt += 1
                kind = "RateLimit" if _is_rate_limit_error(e) else ("Busy" if _is_transient_server_error(e) else "Error")
                print(f"[{kind}] block#{block_idx} attempt {attempt} failed: {type(e).__name__}: {e}")

                msg = f"{type(e).__name__}: {e}".lower()
                if ("403" in msg) or ("insufficient" in msg):
                    print("[AUTH] Remote LLM returned 403 or quota error. Stop retrying this block.")
                    pbar_block.close()
                    pbar_total.close()
                    raise

                _sleep_with_jitter(backoff_base, backoff_factor, attempt - 1, backoff_max, jitter=0.25)

        start = end

    pbar_total.close()
    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser(description="RAGAS + distinct only (centeranchor logic)")
    ap.add_argument("--qa_file", type=str, required=True)
    ap.add_argument("--chunks_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--qa_question_key", type=str, default="question")
    ap.add_argument("--qa_answer_key", type=str, default="answer")
    ap.add_argument("--qa_chunk_id_key", type=str, default="chunk_id")
    ap.add_argument("--qa_chunk_text_key", type=str, default="chunk_text")
    ap.add_argument("--chunks_id_key", type=str, default="id")
    ap.add_argument("--chunks_text_key", type=str, default="content")

    # RAGAS
    ap.add_argument("--metrics", type=str, nargs="+", default=["faithfulness","answer_relevancy"],
                    choices=["faithfulness","answer_relevancy"])
    ap.add_argument("--llm_class", type=str, required=True, help="Import path for the LLM class.")
    ap.add_argument("--llm_kwargs", type=str, default="{}", help="JSON kwargs for the LLM.")
    ap.add_argument("--embedding_class", type=str, required=True, help="Import path for the embeddings class.")
    ap.add_argument("--embedding_kwargs", type=str, default="{}", help="JSON kwargs for the embeddings.")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--chunk_size", type=int, default=32)

    # distinct
    ap.add_argument("--distinct_target", type=str, default="both", choices=["answer","question","both"])
    ap.add_argument("--distinct_n", type=int, default=2, choices=[1,2])
    ap.add_argument("--distinct_group_by", type=str, default="chunk_id")
    ap.add_argument("--lang", type=str, default="auto", choices=["auto","zh","en"])
    ap.add_argument("--token_unit", type=str, default="auto", choices=["auto","char","word"])
    ap.add_argument("--distinct_keep", type=str, nargs="+", default=["content","macro"],
                    choices=["raw","content","conditional","macro"])

    ap.add_argument("--max_retries", type=int, default=12)
    ap.add_argument("--backoff_base", type=float, default=6.0)
    ap.add_argument("--backoff_factor", type=float, default=1.8)
    ap.add_argument("--backoff_max", type=float, default=60.0)

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_any(Path(args.qa_file))
    qa_rows = normalize_qa_rows(loaded)

    chunk_path = args.chunks_file
    chunk_map = None
    if chunk_path:
        chunk_map = load_chunks_map(chunk_path, key_id=args.chunks_id_key, key_text=args.chunks_text_key)
    else:
        chunk_path, chunk_map = auto_discover_chunk_map(
            Path(args.qa_file), qa_rows, args.qa_chunk_id_key, args.chunks_text_key
        )
    if chunk_map is None and chunk_path:
        chunk_map = load_chunks_map(chunk_path, key_id=args.chunks_id_key, key_text=args.chunks_text_key)

    samples=[]; kept=[]
    for it in qa_rows:
        q = resolve_field(it, args.qa_question_key, ["question", "query"])
        a = resolve_field(it, args.qa_answer_key, ["answer", "final_answer"])
        c = resolve_context_from_row(it, chunk_map, args.qa_chunk_id_key, args.qa_chunk_text_key)
        if not (q and a and c): continue
        samples.append(SingleTurnSample(user_input=str(q), retrieved_contexts=[str(c)], response=str(a)))
        kept.append({"id": it.get("id"), "question": str(q), "answer": str(a), "context_text": str(c), args.qa_chunk_id_key: it.get(args.qa_chunk_id_key)})
    if not samples:
        raise SystemExit("No valid samples (missing question/answer/context).")
    df_detail = pd.DataFrame(kept)

    # RAGAS
    llm = _build_component(args.llm_class, args.llm_kwargs)
    llm_wrapper = LangchainLLMWrapper(llm)
    embeddings = _build_component(args.embedding_class, args.embedding_kwargs)
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings)

    metrics=[]
    for name in args.metrics:
        m=_resolve_metric(name)
        if m is None:
            print(f"[WARN] Unsupported ragas metric: {name} (ignored)")
            continue
        metrics.append(m)
    if not metrics:
        raise SystemExit("No valid RAGAS metrics parsed.")

    df_items = run_ragas(samples, metrics, llm_wrapper, emb_wrapper,
                         batch_size=args.batch_size, chunk_size=args.chunk_size,
                         max_retries=args.max_retries, backoff_base=args.backoff_base,
                         backoff_factor=args.backoff_factor, backoff_max=args.backoff_max)

    cols_to_merge=[c for c in df_items.columns if c in ("faithfulness","answer_relevancy")]
    if cols_to_merge:
        df_detail = pd.concat([df_detail, df_items[cols_to_merge].reset_index(drop=True)], axis=1)

    for col in ["faithfulness","answer_relevancy"]:
        if col in df_detail.columns:
            df_detail[col+"_pct"]=df_detail[col].apply(to_pct_raw)

    sample_texts = (df_detail["question"].head(50).astype(str).tolist() +
                    df_detail["answer"].head(50).astype(str).tolist())
    lang, unit = decide_lang_and_unit(sample_texts, args.lang, args.token_unit)
    print(f"[distinct] lang={lang}  token_unit={unit}  keep={','.join(sorted(set(args.distinct_keep)))}")

    summary=[]
    for met in args.metrics:
        if met in df_detail.columns:
            s=pd.to_numeric(df_detail[met], errors="coerce")
            if s.notna().any():
                mean_val=float(s.mean(skipna=True))
                summary.append({"metric":met, "score":mean_val, "score_pct":to_pct_raw(mean_val)})

    # question
    if args.distinct_target in ("question","both"):
        qb=compute_distinct_bundle_centeranchor(
            df_detail, target="question", n=args.distinct_n, group_by=args.distinct_group_by,
            lang=lang, token_unit=unit, keep=args.distinct_keep
        )
        for k,v in qb.items():
            summary.append({"metric":k, "score":float(v), "score_pct":to_pct_raw(v)})

    # answer
    if args.distinct_target in ("answer","both"):
        ab=compute_distinct_bundle_centeranchor(
            df_detail, target="answer", n=args.distinct_n, group_by=args.distinct_group_by,
            lang=lang, token_unit=unit, keep=args.distinct_keep
        )
        for k,v in ab.items():
            summary.append({"metric":k, "score":float(v), "score_pct":to_pct_raw(v)})

    df_detail.to_csv(Path(args.out_dir)/"per_sample_scores.csv", index=False)
    df_detail.to_json(Path(args.out_dir)/"per_sample_scores.json", orient="records", force_ascii=False, indent=2)

    df_sum=pd.DataFrame(summary, columns=["metric","score","score_pct"])
    print("=== Evaluation Summary ==="); print(df_sum.to_string(index=False))
    df_sum.to_csv(Path(args.out_dir)/"summary_metrics.csv", index=False)
    df_sum.to_json(Path(args.out_dir)/"summary_metrics.json", orient="records", force_ascii=False, indent=2)
    print(f"Results saved to {Path(args.out_dir).resolve()}")

if __name__=="__main__":
    main()
