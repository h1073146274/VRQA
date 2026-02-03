#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Basic utilities
def load_any(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

def normalize_qa_rows(loaded) -> List[Dict[str, Any]]:
    if isinstance(loaded, list):
        if loaded and isinstance(loaded[0], dict): return loaded
        raise ValueError("QA list items must be dicts.")
    if isinstance(loaded, dict):
        if isinstance(loaded.get("chunk_results"), list):
            rows = []
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
        rows = []
        for v in loaded.values():
            if isinstance(v, list):
                rows.extend([x for x in v if isinstance(x, dict)])
        if rows: return rows
    raise ValueError("Failed to extract QA rows from JSON.")

def load_chunks_map(chunk_path: Optional[str], key_id="id", key_text="content") -> Optional[Dict[str, str]]:
    if not chunk_path: return None
    loaded = load_any(Path(chunk_path))
    if isinstance(loaded, list):
        out = {}
        for it in loaded:
            if isinstance(it, dict):
                cid = it.get(key_id) or it.get("chunk_id")
                ctext = it.get(key_text) or it.get("chunk_text") or it.get("content")
                if cid and ctext: out[str(cid)] = str(ctext)
        return out
    if isinstance(loaded, dict): return {str(k): str(v) for k, v in loaded.items()}
    raise ValueError("Chunks file must be a list or dict.")

def report_mapping_coverage(qa_rows: List[Dict[str, Any]], chunk_map: Optional[Dict[str, str]], chunk_id_key="chunk_id"):
    if chunk_map is None:
        print("[INFO] chunks_file not provided; rows without chunk_text will be skipped.")
        return
    ids = [str(r.get(chunk_id_key)) for r in qa_rows if r.get(chunk_id_key)]
    uniq = set(ids); hit = sum(1 for cid in uniq if cid in chunk_map)
    print(f"[MAP] unique chunk_ids: {len(uniq)}, hit: {hit}, coverage: {0 if not uniq else hit/len(uniq):.2%}")

def to_pct_raw(x: Optional[float]) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return float(max(0.0, min(100.0, 100.0 * v)))

# UniEval backends
_UNIEVAL_SYS = (
    "You are a strict evaluator. Score the given answer on a specific criterion. "
    "Only output a single integer in [1,5] with no extra text."
)

def _unieval_prompt(dim: str, q: str, a: str, c: Optional[str], with_ctx: bool, lang: str = "auto") -> str:
    if dim == "nat":
        return "Dimension: Naturalness (NAT).\nScore the ANSWER 1-5.\nANSWER\n"+a+"\n"
    if dim == "und":
        return "Dimension: Understandability (UND).\nScore the ANSWER 1-5.\nANSWER\n"+a+"\n"
    if dim == "coh":
        if with_ctx:
            return "Dimension: Coherence (COH).\nGiven QUESTION+CONTEXT, score 1-5.\nQUESTION\n"+q+"\n\nCONTEXT\n"+(c or "")+"\n\nANSWER\n"+a+"\n"
        return "Dimension: Coherence (COH).\nGiven QUESTION only, score 1-5.\nQUESTION\n"+q+"\n\nANSWER\n"+a+"\n"
    return f"Score 1-5: {a}"

def run_unieval_official(df_detail: pd.DataFrame, args):
    """
    Local UniEval backend (offline only).
    - NAT / UND: dialogue task
    - groundedness: summarization consistency proxy
    Notes:
      * monkey-patches sent_tokenize to avoid empty-sentence edge cases
      * fills empty system_output/source with placeholders
    """
    import importlib, torch, os, sys, re
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch.nn as nn

    repo = getattr(args, "unieval_repo", None)
    if repo and os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        utils_mod  = importlib.import_module("unieval.utils")
        eval_mod   = importlib.import_module("unieval.metric.evaluator")
        scorer_mod = importlib.import_module("unieval.metric.scorer")
    except Exception:
        try:
            utils_mod  = importlib.import_module("UniEval.utils")
            eval_mod   = importlib.import_module("UniEval.metric.evaluator")
            scorer_mod = importlib.import_module("UniEval.metric.scorer")
        except Exception:
            utils_mod  = importlib.import_module("utils")
            eval_mod   = importlib.import_module("metric.evaluator")
            scorer_mod = importlib.import_module("metric.scorer")

    convert_to_json = utils_mod.convert_to_json
    get_evaluator   = eval_mod.get_evaluator
    BaseUniEval     = scorer_mod.UniEvaluator

    import re as _re
    def _safe_sent_tokenize(text: str):
        t = (text or "").strip()
        if not t:
            return ["placeholder"]
        sentences = []
        pattern = _re.compile(r"([^.!?;]+[.!?;]?)+")
        for m in pattern.finditer(t):
            s = m.group(0).strip()
            if s:
                sentences.append(s)
        if not sentences:
            sentences = [p.strip() for p in _re.split(r"[\r\n]+", t) if p.strip()]
        if not sentences:
            sentences = [t]
        return sentences
    try:
        eval_mod.sent_tokenize = _safe_sent_tokenize
    except Exception:
        pass

    device_dialog   = getattr(args, "unieval_device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len_dialog  = int(getattr(args, "unieval_input_max_len", 1024) or 1024)
    local_dialog_mp = getattr(args, "unieval_local_path", None)
    if not local_dialog_mp or not os.path.isdir(local_dialog_mp):
        raise RuntimeError(f"--unieval_local_path does not exist or is not a directory: {local_dialog_mp}")

    local_sum_mp = getattr(args, "unieval_sum_local_path", None)
    if not local_sum_mp or not os.path.isdir(local_sum_mp):
        raise RuntimeError(f"--unieval_sum_local_path does not exist or is not a directory: {local_sum_mp}")

    class _LocalUniEvaluatorDialog(BaseUniEval):
        def __init__(self, model_name_or_path=None, max_length=1024, device='cuda:0', cache_dir=None):
            self.device = device
            self.max_length = max_length
            self.config = AutoConfig.from_pretrained(local_dialog_mp, cache_dir=cache_dir, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(local_dialog_mp, cache_dir=cache_dir, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                local_dialog_mp, config=self.config, cache_dir=cache_dir, local_files_only=True
            )
            self.model.eval().to(device)
            self.softmax = nn.Softmax(dim=1)
            self.pos_id = self.tokenizer("Yes")["input_ids"][0]
            self.neg_id = self.tokenizer("No")["input_ids"][0]

    class _LocalUniEvaluatorSum(BaseUniEval):
        def __init__(self, model_name_or_path=None, max_length=1024, device='cuda:0', cache_dir=None):
            self.device = device
            self.max_length = max_length
            self.config = AutoConfig.from_pretrained(local_sum_mp, cache_dir=cache_dir, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(local_sum_mp, cache_dir=cache_dir, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                local_sum_mp, config=self.config, cache_dir=cache_dir, local_files_only=True
            )
            self.model.eval().to(device)
            self.softmax = nn.Softmax(dim=1)
            self.pos_id = self.tokenizer("Yes")["input_ids"][0]
            self.neg_id = self.tokenizer("No")["input_ids"][0]

    scorer_mod.UniEvaluator = _LocalUniEvaluatorDialog
    eval_mod.UniEvaluator   = _LocalUniEvaluatorDialog
    dialog_evaluator = get_evaluator("dialogue", max_length=max_len_dialog, device=device_dialog, cache_dir=None)

    q_key = getattr(args, "qa_question_key", "question")
    a_key = getattr(args, "qa_answer_key", "answer")
    src_list = []
    output_list = []
    for _, row in df_detail.iterrows():
        q = str(row.get(q_key, "") or "")
        a = str(row.get(a_key, "") or "")
        src_list.append((q.strip() + "\n\n") if q else "")
        output_list.append(a)

    data_dialog = convert_to_json(output_list=output_list, src_list=src_list)

    dialog_scores = dialog_evaluator.evaluate(
        data_dialog,
        dims=["naturalness", "understandability"],
        overall=False,
        print_result=False
    )
    nat = [float(s.get("naturalness", 0.0)) for s in dialog_scores]
    und = [float(s.get("understandability", 0.0)) for s in dialog_scores]

    device_sum  = device_dialog
    max_len_sum = int(getattr(args, "unieval_sum_max_len", 1536) or 1536)
    scorer_mod.UniEvaluator = _LocalUniEvaluatorSum
    eval_mod.UniEvaluator   = _LocalUniEvaluatorSum
    sum_evaluator = get_evaluator("summarization", max_length=max_len_sum, device=device_sum, cache_dir=None)

    c_key = getattr(args, "qa_chunk_text_key", "context_text")

    _sent_re = re.compile(r"(.*?)[.!?;]\s*")
    def _split_sents(txt: str):
        sents = [m.group(1) for m in _sent_re.finditer(txt)]
        tail = _sent_re.sub("", txt).strip()
        if tail: sents.append(tail)
        return [s.strip() for s in sents if s.strip()]

    def _overlap_score(s: str, a: str):
        def _tok(t):
            if re.search(r"[\u4e00-\u9fff]", t):
                return [ch for ch in t if not ch.isspace()]
            return re.findall(r"[A-Za-z0-9]+", t.lower())
        st = set(_tok(s)); sa = set(_tok(a))
        return len(st & sa)

    def _make_ctx_snippet(ctx: str, ans: str, max_chars: int = 1500):
        if not ctx: return ""
        sents = _split_sents(ctx)
        sents = sorted(sents, key=lambda s: _overlap_score(s, ans), reverse=True)
        buf, total = [], 0
        for s in sents:
            if total + len(s) + 1 > max_chars: break
            buf.append(s); total += len(s) + 1
        return ". ".join(buf)

    output_list_sum = [(str(o).strip() if str(o).strip() else "placeholder.") for o in output_list]

    src_list_sum = []
    for (_, row), a in zip(df_detail.iterrows(), output_list_sum):
        c = str(row.get(c_key, "") or "")
        snippet = _make_ctx_snippet(c, a, max_chars=max_len_sum - 100)
        src_list_sum.append(snippet if snippet.strip() else "placeholder.")

    data_sum = convert_to_json(output_list=output_list_sum, src_list=src_list_sum)

    sum_scores = sum_evaluator.evaluate(
        data_sum,
        dims=["consistency"],
        overall=False,
        print_result=False
    )
    grd = [float(s.get("consistency", 0.0)) for s in sum_scores]

    out = df_detail.copy()
    out["unieval_nat"] = nat
    out["unieval_und"] = und
    out["unieval_groundedness"] = grd

    means = {
        "unieval_nat_mean": float(np.mean(nat)) if nat else 0.0,
        "unieval_und_mean": float(np.mean(und)) if und else 0.0,
        "unieval_groundedness_mean": float(np.mean(grd)) if grd else 0.0,
    }
    return out, means

def _dtype_from_str(name: str):
    import torch
    name=(name or "auto").lower()
    if name=="auto": return "auto"
    if name=="float16": return torch.float16
    if name=="bfloat16": return torch.bfloat16
    if name=="float32": return torch.float32
    return "auto"

def _load_hf_unieval_model(path: str, device: Optional[str], dtype_str: str):
    import torch, os
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Local evaluator model path does not exist: {path}")
    dtype = _dtype_from_str(dtype_str)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model=None
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=(dtype if dtype!="auto" else None), local_files_only=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=(dtype if dtype!="auto" else None), local_files_only=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if getattr(model.config,"pad_token_id",None) is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    if getattr(model.config,"is_encoder_decoder",False) and getattr(model.config,"decoder_start_token_id",None) is None:
        model.config.decoder_start_token_id = getattr(tok,"bos_token_id",None) or tok.eos_token_id or model.config.pad_token_id
    if device: model = model.to(device)
    else:
        import torch
        if torch.cuda.is_available(): model = model.to("cuda")
    model.eval()
    return tok, model

def _hf_first_token_logits(prompts: List[str], tok, model, *, batch_size=8, input_max_len=1024):
    import torch
    all_logits=[]
    dev=model.device
    for i in range(0,len(prompts),batch_size):
        batch=prompts[i:i+batch_size]
        enc=tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=input_max_len).to(dev)
        try:
            out=model.generate(**enc, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
            step=out.scores[0]
            for r in range(step.shape[0]):
                all_logits.append(step[r].detach().cpu())
        except Exception:
            outputs=model(**enc, use_cache=False)
            logits=outputs.logits
            for r in range(logits.shape[0]):
                all_logits.append(logits[r,-1,:].detach().cpu())
    return all_logits

def _map_logits_to_unit(tok, logits_batch):
    import torch
    base_digits=["1","2","3","4","5"]
    word_digits=["one", "two", "three", "four", "five"]
    prefixes=["", " "]
    cand_map={}
    for i,d in enumerate(base_digits, start=1):
        for pref in prefixes:
            ids = tok.encode(pref+d, add_special_tokens=False)
            if len(ids)==1: cand_map[ids[0]]=i
    for i, d in enumerate(word_digits, start=1):
        for pref in prefixes:
            ids = tok.encode(pref+d, add_special_tokens=False)
            if len(ids)==1: cand_map[ids[0]]=i
    if not cand_map:
        return [0.0]*len(logits_batch)
    tids=torch.tensor(list(cand_map.keys()))
    n_of_tid=[cand_map[t.item()] for t in tids]
    scores=[]
    for logit in logits_batch:
        sub=logit[tids]
        prob=torch.softmax(sub, dim=-1)
        j=int(torch.argmax(prob).item())
        scores.append((n_of_tid[j]-1)/4.0)
    return scores

def run_unieval_hf(df_detail: pd.DataFrame, args):
    with_ctx = bool(getattr(args,"coh_with_context",False))
    lang_opt = getattr(args,"lang","auto")
    q_key = getattr(args,"qa_question_key","question")
    a_key = getattr(args,"qa_answer_key","answer")
    c_key = getattr(args,"qa_chunk_text_key",None)
    if not c_key or c_key not in df_detail.columns:
        c_key = "context_text" if "context_text" in df_detail.columns else "chunk_text"
    nat_prompts,und_prompts,coh_prompts=[],[],[]
    for _,row in df_detail.iterrows():
        q=str(row.get(q_key,"") or ""); a=str(row.get(a_key,"") or ""); c=str(row.get(c_key,"") or "")
        nat_prompts.append(_unieval_prompt("nat",q,a,None,False,lang_opt))
        und_prompts.append(_unieval_prompt("und",q,a,None,False,lang_opt))
        coh_prompts.append(_unieval_prompt("coh",q,a,(c if with_ctx else None),with_ctx,lang_opt))
    tok,mdl=_load_hf_unieval_model(
        getattr(args,"unieval_local_path",None),
        getattr(args,"unieval_device",None),
        getattr(args,"unieval_dtype","auto"),
    )
    nat=_map_logits_to_unit(tok,_hf_first_token_logits(nat_prompts,tok,mdl,batch_size=getattr(args,"unieval_batch_size",8),input_max_len=getattr(args,"unieval_input_max_len",1024)))
    und=_map_logits_to_unit(tok,_hf_first_token_logits(und_prompts,tok,mdl,batch_size=getattr(args,"unieval_batch_size",8),input_max_len=getattr(args,"unieval_input_max_len",1024)))
    coh=_map_logits_to_unit(tok,_hf_first_token_logits(coh_prompts,tok,mdl,batch_size=getattr(args,"unieval_batch_size",8),input_max_len=getattr(args,"unieval_input_max_len",1024)))
    out=df_detail.copy(); out["unieval_nat"]=nat; out["unieval_und"]=und; out["unieval_coh"]=coh
    means={"unieval_nat_mean":float(np.mean(nat)) if nat else 0.0,
           "unieval_und_mean":float(np.mean(und)) if und else 0.0,
           "unieval_coh_mean":float(np.mean(coh)) if coh else 0.0}
    return out, means

def main():
    ap = argparse.ArgumentParser(description="UniEval-only runner")
    ap.add_argument("--qa_file", type=str, required=True)
    ap.add_argument("--chunks_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--qa_question_key", type=str, default="question")
    ap.add_argument("--qa_answer_key", type=str, default="answer")
    ap.add_argument("--qa_chunk_id_key", type=str, default="chunk_id")
    ap.add_argument("--qa_chunk_text_key", type=str, default="context_text")

    ap.add_argument("--coh_with_context", action="store_true")
    ap.add_argument("--lang", type=str, default="auto", choices=["auto","zh","en"])
    ap.add_argument("--enable_unieval", action="store_true", default=True)
    ap.add_argument("--unieval_backend", type=str, default="official", choices=["official","hf"])
    ap.add_argument("--unieval_input_max_len", type=int, default=1024)
    ap.add_argument("--unieval_device", type=str, default=None)
    ap.add_argument("--unieval_dtype", type=str, default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--unieval_batch_size", type=int, default=8)
    ap.add_argument("--unieval_local_path", type=str, default=None)
    ap.add_argument("--unieval_repo", type=str, default=None)
    ap.add_argument("--unieval_sum_local_path", type=str, default="", help="Local summarization checkpoint directory.")
    ap.add_argument("--unieval_sum_max_len", type=int, default=1536, help="Max input length for summarization backend.")


    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_any(Path(args.qa_file))
    qa_rows = normalize_qa_rows(loaded)
    chunk_map = load_chunks_map(args.chunks_file)
    report_mapping_coverage(qa_rows, chunk_map, chunk_id_key=args.qa_chunk_id_key)

    rows = []
    for it in qa_rows:
        q = str(it.get(args.qa_question_key,"") or "")
        a = str(it.get(args.qa_answer_key,"") or "")
        ctx = it.get(args.qa_chunk_text_key) or (chunk_map.get(str(it.get(args.qa_chunk_id_key))) if chunk_map else None)
        if not (q and a and ctx): continue
        rows.append({"id": it.get("id"), "question": q, "answer": a, "context_text": str(ctx)})

    if not rows:
        raise SystemExit("No valid samples (missing question/answer/context).")
    df = pd.DataFrame(rows)

    backend = args.unieval_backend
    if backend == "official":
        df_out, means = run_unieval_official(df, args)
    else:
        df_out, means = run_unieval_hf(df, args)

    for col in ["unieval_nat", "unieval_und", "unieval_groundedness"]:
        if col in df_out.columns:
            df_out[col + "_pct"] = df_out[col].apply(to_pct_raw)

    df_out.to_csv(out_dir/"per_sample_scores.csv", index=False)
    df_out.to_json(out_dir/"per_sample_scores.json", orient="records", force_ascii=False, indent=2)

    summary_rows = [{"metric":k, "score":float(v), "score_pct":to_pct_raw(v)} for k,v in means.items()]
    df_sum = pd.DataFrame(summary_rows, columns=["metric","score","score_pct"])
    print("=== Evaluation Summary ==="); print(df_sum.to_string(index=False))

    df_sum.to_csv(out_dir/"summary_metrics.csv", index=False)
    df_sum.to_json(out_dir/"summary_metrics.json", orient="records", force_ascii=False, indent=2)
    print(f"Results saved to {out_dir.resolve()}")

if __name__=="__main__":
    main()
