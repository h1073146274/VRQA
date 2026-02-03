from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import requests

from vrqa.prompts.qa_prompts import (
    build_answer_generation_prompt,
    build_qa_pair_scoring_prompt,
    build_question_generation_prompt,
)
from vrqa.router.features import build_chunk_context_features
from vrqa.router.linucb import LinUCBRouter, ucb_gains_from_pairs
from vrqa.selector.submodular import SubmodularSelector, AdaptiveSelectorController
from vrqa.utils.similarity_utils import compute_similarity, compute_max_sentence_sim
from vrqa.data.anchors import LlmConfig, load_llm_config_from_env


VIEWS = [
    "definition",
    "mechanism",
    "application",
    "comparison",
    "cause",
    "effect",
    "condition",
    "example",
    "quantitative",
    "temporal",
]

_SCORER_MODEL = None
_SCORER_TOKENIZER = None
_SCORER_DEVICE = "cpu"
_SCORER_MODEL_PATH = ""
_SCORER_WARNED = False


# -----------------------------
# LLM helpers
# -----------------------------


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


def call_llm(prompt: str, *, llm: LlmConfig, delay: float = 0.0) -> str:
    url = _resolve_chat_url(llm.api_base_url)
    headers = {
        "Authorization": f"Bearer {llm.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=llm.timeout_sec)
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = {"content": resp.text or ""}
    if delay > 0:
        time.sleep(delay)
    return _extract_text_from_response(resp_json)


def _safe_json_array(raw: str) -> List[Any]:
    if not raw:
        return []
    raw = raw.strip()
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else []
    except Exception:
        pass
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _safe_json_obj(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def set_scorer_model_path(path: Optional[str]) -> None:
    global _SCORER_MODEL_PATH
    _SCORER_MODEL_PATH = (path or "").strip()


def _ensure_local_scorer() -> bool:
    global _SCORER_MODEL, _SCORER_TOKENIZER, _SCORER_DEVICE, _SCORER_WARNED, _SCORER_MODEL_PATH
    if _SCORER_MODEL is not None and _SCORER_TOKENIZER is not None:
        return True
    if not _SCORER_MODEL_PATH:
        return False

    scorer_path = _SCORER_MODEL_PATH
    try:
        if os.path.isdir(scorer_path):
            files = os.listdir(scorer_path)
            has_tokenizer = any(
                fn in files for fn in ("tokenizer_config.json", "tokenizer.json", "config.json")
            )
            subdirs = [f for f in files if os.path.isdir(os.path.join(scorer_path, f))]
            if (not has_tokenizer) and len(subdirs) == 1:
                inner = os.path.join(scorer_path, subdirs[0])
                inner_files = os.listdir(inner)
                if any(fn in inner_files for fn in ("tokenizer_config.json", "tokenizer.json", "config.json")):
                    scorer_path = inner
                    _SCORER_MODEL_PATH = scorer_path
    except Exception:
        scorer_path = _SCORER_MODEL_PATH

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        if not _SCORER_WARNED:
            _SCORER_WARNED = True
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(scorer_path)
        model = AutoModelForCausalLM.from_pretrained(scorer_path)
        if torch.cuda.is_available():
            device = "cuda"
            model.to(device)
        else:
            device = "cpu"
            model.to(device)
        model.eval()
        _SCORER_MODEL = model
        _SCORER_TOKENIZER = tokenizer
        _SCORER_DEVICE = device
        return True
    except Exception:
        if not _SCORER_WARNED:
            _SCORER_WARNED = True
        _SCORER_MODEL = None
        _SCORER_TOKENIZER = None
        return False


def score_with_local_scorer(prompt: str, default: Optional[Dict[str, Any]] = None, max_new_tokens: int = 256) -> Dict[str, Any]:
    if default is None:
        default = {"total_score": 0.0, "dimension_scores": {}, "comment": ""}
    if not _ensure_local_scorer():
        return default
    try:
        import torch  # type: ignore
    except Exception:
        return default

    tokenizer = _SCORER_TOKENIZER
    model = _SCORER_MODEL
    if tokenizer is None or model is None:
        return default

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(_SCORER_DEVICE)  # type: ignore

        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0}
            pad_id = getattr(tokenizer, "pad_token_id", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None and eos_id is not None:
                gen_kwargs["pad_token_id"] = eos_id
            outputs = model.generate(**inputs, **gen_kwargs)  # type: ignore

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0, input_len:]
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
        obj = _safe_json_obj(text_out)
        if not isinstance(obj, dict):
            return default
        try:
            obj["total_score"] = float(obj.get("total_score", 0.0))
        except Exception:
            obj["total_score"] = 0.0
        obj.setdefault("dimension_scores", {})
        obj.setdefault("comment", "")
        return obj
    except Exception:
        return default


# -----------------------------
# Core generation logic
# -----------------------------


def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    return [s.strip() for s in sents if s.strip()]


def extract_candidates(
    content: str,
    *,
    llm: Optional[LlmConfig],
    candidate_count: int,
    api_delay: float,
) -> List[Dict[str, Any]]:
    if not content:
        return [{"candidate_id": 0, "candidate_answer": ""}]

    if llm is None:
        # fallback: first N sentences
        sents = _split_sentences(content)
        picks = sents[: max(1, candidate_count)]
        if not picks:
            picks = [content[:200]]
        return [
            {"candidate_id": i, "candidate_answer": s, "importance_rank": i + 1}
            for i, s in enumerate(picks)
        ]

    prompt = (
        f"Extract {candidate_count} DISTINCT factual information points from the Body, ordered by importance (highest first).\n"
        "Return strictly as a JSON array of strings.\n\n"
        f"[Body]\n{content}\n\n"
        f"Now extract {candidate_count} candidate answers in descending importance:"
    )
    raw = call_llm(prompt, llm=llm, delay=api_delay)
    arr = [str(x).strip() for x in _safe_json_array(raw) if str(x).strip()]
    if not arr:
        sents = _split_sentences(content)
        arr = sents[: max(1, candidate_count)]
    return [
        {"candidate_id": i, "candidate_answer": s, "importance_rank": i + 1}
        for i, s in enumerate(arr[: max(1, candidate_count)])
    ]


def generate_candidates_for_chunk(
    chunk: Dict[str, Any],
    chosen_views: List[str],
    pick_per_view: int,
    *,
    llm: LlmConfig,
    api_delay: float,
) -> List[Dict[str, Any]]:
    content = str(chunk.get("content") or chunk.get("text") or "")
    meta = chunk.get("meta") or {}
    chunk_summary = str(chunk.get("summary") or meta.get("chunk_summary") or meta.get("summary") or "")
    chunk_keywords_raw = (
        chunk.get("paragraph_keywords")
        or meta.get("chunk_keywords")
        or meta.get("paragraph_keywords")
        or []
    )
    chunk_keywords = list(chunk_keywords_raw) if isinstance(chunk_keywords_raw, list) else []

    article_abstract = str(meta.get("abstract") or "")
    article_keywords = meta.get("keywords") or []

    candidate_answer = str(chunk.get("candidate_answer") or meta.get("candidate_answer") or "")

    items: List[Dict[str, Any]] = []
    existing_questions: List[str] = []

    for v in chosen_views:
        q_prompt = build_question_generation_prompt(
            text=content,
            abstract=article_abstract,
            keywords=article_keywords,
            candidate_answer=candidate_answer,
            number=pick_per_view,
            attempt_round=1,
            view=v,
            existing_questions=existing_questions,
            chunk_summary=chunk_summary,
            chunk_keywords=chunk_keywords,
        )
        q_raw = call_llm(q_prompt, llm=llm, delay=api_delay)
        questions = [str(q).strip() for q in _safe_json_array(q_raw) if str(q).strip()]
        for q in questions:
            existing_questions.append(q)
            items.append(
                {
                    "question": q,
                    "meta": {"view": v},
                    "anchor": {"align": 0.0, "bridge": 0.0, "conflict": 0.0},
                }
            )

    return items


def infer_on_items(items: List[Dict[str, Any]], *, chunk: Dict[str, Any], llm: LlmConfig, api_delay: float) -> List[Dict[str, Any]]:
    content = str(chunk.get("content") or chunk.get("text") or "")
    candidate_answer = str(chunk.get("candidate_answer") or (chunk.get("meta") or {}).get("candidate_answer") or "")

    out: List[Dict[str, Any]] = []
    for it in items:
        q = str(it.get("question") or "")
        if not q:
            continue
        prompt_final = build_answer_generation_prompt(q, content, candidate_answer)
        raw = call_llm(prompt_final, llm=llm, delay=api_delay)
        ans_list = [str(a).strip() for a in _safe_json_array(raw) if str(a).strip()]
        if not ans_list:
            ans_list = [""]
        # pick best answer by similarity to candidate_answer
        best = {"ans": ans_list[0], "sim": -1.0}
        for a in ans_list:
            sim = compute_similarity(a, candidate_answer) if candidate_answer else 0.0
            if sim > best["sim"]:
                best = {"ans": a, "sim": sim}
        it = dict(it)
        it["answer"] = best["ans"]
        it["similarity_with_candidate"] = float(best["sim"])
        it["similarity_detail"] = {
            "sim_to_candidate": float(best["sim"]),
            "sim_to_sentence": float(compute_max_sentence_sim(best["ans"], content) if best["ans"] else 0.0),
        }
        out.append(it)

    return out


def score_items_with_scorer(
    items: List[Dict[str, Any]],
    *,
    chunk: Dict[str, Any],
    llm: LlmConfig,
    api_delay: float,
    min_score: float,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    meta = chunk.get("meta") or {}
    content = str(chunk.get("content") or chunk.get("text") or "")
    abstract = str(meta.get("abstract") or "")
    keywords = meta.get("keywords") or []

    scored: List[Dict[str, Any]] = []
    for it in items:
        q = str(it.get("question") or "")
        a = str(it.get("answer") or "")
        if not q or not a:
            continue

        prompt = build_qa_pair_scoring_prompt(q, a, content, abstract, keywords)
        if _SCORER_MODEL_PATH:
            obj = score_with_local_scorer(prompt, default={"total_score": 0.0, "dimension_scores": {}, "comment": ""})
        else:
            raw = call_llm(prompt, llm=llm, delay=api_delay)
            obj = _safe_json_obj(raw)

        total_score = float(obj.get("total_score", 0.0) or 0.0)
        total_score = max(0.0, min(10.0, total_score))
        it = dict(it)
        it["total_score"] = total_score
        it["ans_score"] = total_score / 10.0
        it["score_detail"] = obj

        sim_c = float(it.get("similarity_with_candidate", 0.0) or 0.0)
        sim_d = it.get("similarity_detail") or {}
        sim_s = float(sim_d.get("sim_to_sentence", 0.0) or 0.0) if isinstance(sim_d, dict) else 0.0
        if total_score < float(min_score):
            continue
        if sim_c < float(similarity_threshold) and sim_s < float(similarity_threshold):
            continue

        scored.append(it)

    return scored


# -----------------------------
# Main per-chunk flow
# -----------------------------


def run_one_chunk(
    chunk: Dict[str, Any],
    *,
    router: LinUCBRouter,
    selector: SubmodularSelector,
    K: int,
    B: int,
    pick_per_view: int,
    stage: int,
    llm: LlmConfig,
    api_delay: float,
    min_score: float,
    similarity_threshold: float,
    stage1_uniform_views: bool,
    stage1_uniform_prob: float,
    adaptive: Optional[AdaptiveSelectorController] = None,
) -> Dict[str, Any]:
    meta = chunk.get("meta") or {}
    content = str(chunk.get("content") or chunk.get("text") or "")

    chunk_summary = str(chunk.get("summary") or meta.get("chunk_summary") or meta.get("summary") or "")
    chunk_keywords_raw = (
        chunk.get("paragraph_keywords")
        or meta.get("chunk_keywords")
        or meta.get("paragraph_keywords")
        or []
    )
    chunk_keywords = list(chunk_keywords_raw) if isinstance(chunk_keywords_raw, list) else []

    article_meta = {
        "title": str(chunk.get("title") or meta.get("title") or ""),
        "abstract": str(meta.get("abstract") or ""),
        "keywords": meta.get("keywords") or [],
    }

    candidate_answer = str(chunk.get("candidate_answer") or meta.get("candidate_answer") or "")

    x_ctx, _d = build_chunk_context_features(
        content=content,
        chunk_summary=chunk_summary,
        chunk_keywords=chunk_keywords,
        article_meta=article_meta,
        candidate_answer=candidate_answer,
    )

    chosen_views: List[str] = []
    is_stage1_mode = (
        stage1_uniform_views
        and (not getattr(router, "train_linucb", True))
        and bool(getattr(router, "train_film", False))
    )
    if is_stage1_mode:
        p_uniform = max(0.0, min(1.0, float(stage1_uniform_prob)))
        if random.random() < p_uniform:
            try:
                chosen_views = random.sample(VIEWS, k=K)
            except Exception:
                chosen_views = VIEWS[:K]
        else:
            chosen_views = router.select(x_ctx, k=K)
    else:
        chosen_views = router.select(x_ctx, k=K)

    if adaptive is not None:
        quotas = adaptive.suggest_view_quota(chosen_views, pick_per_view)
        pick_per_view_eff = max(1, int(sum(quotas.values()) / max(1, len(quotas))))
    else:
        pick_per_view_eff = pick_per_view

    items = generate_candidates_for_chunk(
        chunk,
        chosen_views,
        pick_per_view_eff,
        llm=llm,
        api_delay=api_delay,
    )
    items = infer_on_items(items, chunk=chunk, llm=llm, api_delay=api_delay)
    items = score_items_with_scorer(
        items,
        chunk=chunk,
        llm=llm,
        api_delay=api_delay,
        min_score=min_score,
        similarity_threshold=similarity_threshold,
    )

    selected = selector.select(items, M=B)

    if int(stage) == 2 and selected:
        gains = ucb_gains_from_pairs(selected)
        for v, g in gains.items():
            reward = float(g) / max(
                1.0,
                sum(1 for it in selected if (it.get("meta") or {}).get("view") == v),
            )
            router.update(v, x_ctx, reward)

    if adaptive is not None:
        accepted_views = list({(it.get("meta") or {}).get("view", "unknown") for it in selected})
        adaptive.update_view_outcome(chosen_views, accepted_views)
        adaptive.update_selector_success(float(len(selected)) / max(1.0, float(B)))

    return {
        "chunk_id": (chunk.get("id") or meta.get("chunk_id")),
        "chosen_views": chosen_views,
        "num_items": len(items),
        "num_selected": len(selected),
        "selected": selected,
    }


# -----------------------------
# Full pipeline
# -----------------------------


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return [x for x in obj["data"] if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_ndjson(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _save_ucb_state(path: str, router: LinUCBRouter) -> None:
    payload = {"linucb": router.get_state()}
    _save_json(path, payload)


def _load_ucb_state(path: str, router: LinUCBRouter) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("linucb"), dict):
            return bool(router.load_state(obj["linucb"]))
    except Exception:
        return False
    return False


def _summarize_view_stats(selected_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    gains = ucb_gains_from_pairs(selected_pairs)
    counts: Dict[str, int] = {}
    for it in selected_pairs:
        v = (it.get("meta") or {}).get("view", "unknown")
        counts[v] = counts.get(v, 0) + 1
    reward_mean = {
        v: (float(gains.get(v, 0.0)) / max(1.0, float(counts.get(v, 0))))
        for v in counts.keys()
    }
    return {"view_selection_counts": counts, "view_reward_mean": reward_mean}


def main() -> None:
    ap = argparse.ArgumentParser("VRQA generator")
    ap.add_argument("--input", required=True, help="Input chunks JSON or JSONL.")
    ap.add_argument("--output", required=True, help="Output path for QA results (JSON).")
    ap.add_argument("--num_rounds", type=int, default=1)
    ap.add_argument("--sample_ratio_per_round", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--round_offset", type=int, default=0)
    ap.add_argument("--enable_film", type=str, default="true")
    ap.add_argument("--train_linucb", type=str, default="true")
    ap.add_argument("--train_film", type=str, default="true")
    ap.add_argument("--ucb_exploit_only", type=str, default="false")
    ap.add_argument("--pick_per_view", type=int, default=3)
    ap.add_argument("--selector_m", type=int, default=3)
    ap.add_argument("--router_k", type=int, default=3)
    ap.add_argument("--chars_per_candidate", type=int, default=240)
    ap.add_argument("--api_delay", type=float, default=0.0)
    ap.add_argument("--similarity_analysis", default="", help="Optional output path (not used, kept for compat).")
    ap.add_argument("--freeze_candidates_after_first", type=str, default="false")
    ap.add_argument("--ucb_state", default="", help="Path to persist LinUCB state (JSON).")
    ap.add_argument("--ucb_metrics", default="", help="Path to persist LinUCB metrics (NDJSON).")
    ap.add_argument("--ucb_viewlog_combined", default="", help="Path to append viewlog (NDJSON).")
    ap.add_argument("--mmr_mode", type=str, default="true")
    ap.add_argument("--mmr_lambda", type=float, default=0.2)
    ap.add_argument("--scorer_model_path", default="", help="Local scorer model path.")
    ap.add_argument("--min_score", type=float, default=6.0)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--stage1_uniform_views", type=str, default="false")
    ap.add_argument("--stage1_uniform_prob", type=float, default=0.5)
    args = ap.parse_args()

    random.seed(args.random_seed)

    llm = load_llm_config_from_env()
    if llm is None:
        raise RuntimeError("LLM_API_KEY/LLM_API_BASE_URL not set; cannot run generation.")

    chunks = _load_json_or_jsonl(args.input)
    if not chunks:
        raise RuntimeError("No chunks loaded from input.")

    router = LinUCBRouter(
        arms=VIEWS,
        d=32,
        alpha=1.4,
        enable_film=(str(args.enable_film).lower() == "true"),
        train_linucb=(str(args.train_linucb).lower() == "true"),
        train_film=(str(args.train_film).lower() == "true"),
    )
    selector = SubmodularSelector(
        mmr_mode=(str(args.mmr_mode).lower() == "true"),
        mmr_lambda=float(args.mmr_lambda),
    )
    adaptive = AdaptiveSelectorController()

    if str(args.ucb_exploit_only).lower() == "true":
        router.alpha = 0.0

    if args.scorer_model_path:
        set_scorer_model_path(args.scorer_model_path)

    if args.ucb_state:
        _load_ucb_state(args.ucb_state, router)

    total_rounds = int(max(1, args.num_rounds))
    all_round_outputs: List[Dict[str, Any]] = []
    frozen_candidates_by_chunk: Dict[str, List[Dict[str, Any]]] = {}

    for r in range(1, total_rounds + 1):
        stage = 2 if (str(args.train_linucb).lower() == "true" and str(args.train_film).lower() == "false") else 1
        # sample chunks
        ratio = max(0.0, min(1.0, float(args.sample_ratio_per_round)))
        if ratio < 1.0:
            sample_count = max(1, int(len(chunks) * ratio))
            chunk_batch = random.sample(chunks, k=sample_count)
        else:
            chunk_batch = chunks

        chunk_results = []
        all_selected_pairs: List[Dict[str, Any]] = []
        for ch in chunk_batch:
            content = str(ch.get("content") or ch.get("text") or "")
            if not content:
                continue

            chunk_id = str(ch.get("id") or (ch.get("meta") or {}).get("chunk_id") or "")
            use_frozen = (r >= 2) and (str(args.freeze_candidates_after_first).lower() == "true")
            if use_frozen and chunk_id in frozen_candidates_by_chunk:
                candidates = frozen_candidates_by_chunk[chunk_id]
            else:
                candidate_count = max(1, len(content) // max(1, int(args.chars_per_candidate)))
                candidates = extract_candidates(
                    content,
                    llm=llm,
                    candidate_count=candidate_count,
                    api_delay=args.api_delay,
                )

            accepted_results: List[Dict[str, Any]] = []
            for cand in candidates:
                ch2 = dict(ch)
                meta = dict(ch.get("meta") or {})
                meta["candidate_answer"] = cand.get("candidate_answer", "")
                ch2["meta"] = meta
                ch2["candidate_answer"] = cand.get("candidate_answer", "")

                one = run_one_chunk(
                    ch2,
                    router=router,
                    selector=selector,
                    K=min(int(args.router_k), len(VIEWS)),
                    B=int(args.selector_m),
                    pick_per_view=int(args.pick_per_view),
                    stage=stage,
                    llm=llm,
                    api_delay=float(args.api_delay),
                    min_score=float(args.min_score),
                    similarity_threshold=float(args.threshold),
                    stage1_uniform_views=(str(args.stage1_uniform_views).lower() == "true"),
                    stage1_uniform_prob=float(args.stage1_uniform_prob),
                    adaptive=adaptive,
                )

                for it in one.get("selected", []):
                    it = dict(it)
                    it["candidate_answer"] = cand.get("candidate_answer", "")
                    it["candidate_id"] = cand.get("candidate_id", 0)
                    it["importance_rank"] = cand.get("importance_rank", 1)
                    it["final_answer"] = it.get("answer", "")
                    accepted_results.append(it)
                    all_selected_pairs.append(it)

            chunk_results.append(
                {
                    "chunk_id": chunk_id or ch.get("id") or (ch.get("meta") or {}).get("chunk_id"),
                    "articleId": ch.get("articleId"),
                    "accepted_results": accepted_results,
                }
            )

        round_payload = {
            "round": r + int(args.round_offset),
            "chunk_results": chunk_results,
        }
        all_round_outputs.append(round_payload)

        # freeze candidates after first round
        if r == 1 and (str(args.freeze_candidates_after_first).lower() == "true"):
            for ch in chunk_batch:
                cid = str(ch.get("id") or (ch.get("meta") or {}).get("chunk_id") or "")
                if not cid:
                    continue
                if cid in frozen_candidates_by_chunk:
                    continue
                content = str(ch.get("content") or ch.get("text") or "")
                if not content:
                    continue
                candidate_count = max(1, len(content) // max(1, int(args.chars_per_candidate)))
                frozen_candidates_by_chunk[cid] = extract_candidates(
                    content,
                    llm=llm,
                    candidate_count=candidate_count,
                    api_delay=args.api_delay,
                )

        # per-round outputs (compat with .roundN.json)
        base = args.output
        if total_rounds > 1:
            round_path = os.path.splitext(base)[0] + f".round{r}.json"
        else:
            round_path = base
        _save_json(round_path, round_payload)

        # also write a rolling combined output
        if total_rounds > 1:
            _save_json(args.output, {"rounds": all_round_outputs})
        else:
            _save_json(args.output, round_payload)

        # metrics + viewlog
        stats = _summarize_view_stats(all_selected_pairs)
        if args.ucb_metrics:
            _append_ndjson(args.ucb_metrics, {"round": round_payload["round"], **stats})
        if args.ucb_viewlog_combined:
            _append_ndjson(args.ucb_viewlog_combined, {"round": round_payload["round"], **stats})

        # persist router state per round
        if args.ucb_state:
            _save_ucb_state(args.ucb_state, router)

    print(f"[OK] wrote results -> {args.output}")


if __name__ == "__main__":
    main()
