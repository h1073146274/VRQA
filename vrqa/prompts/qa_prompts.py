"""Prompt construction utilities for QA generation."""

from typing import List

from vrqa.utils.similarity_utils import extract_representative_keywords


def build_question_generation_prompt(
    text: str,
    abstract: str,
    keywords: List[str],
    candidate_answer: str,
    number: int = 3,
    attempt_round: int = 1,
    view: str = None,
    existing_questions: List[str] = None,
    *,
    chunk_summary: str = "",
    chunk_keywords: List[str] = None,
) -> str:
    """Create question-generation prompt with optional view guidance."""

    chunk_keywords = chunk_keywords or []
    rep_kws = extract_representative_keywords(text, candidate_answer) or []
    guide_kws = list(dict.fromkeys([*rep_kws, *(keywords or []), *chunk_keywords]))[:12]
    kw_hint = ", ".join([k for k in guide_kws if k.strip()]) if guide_kws else "N/A"

    angle_strategies = {
        "definition": "Ask from the perspective of definition/boundary.",
        "mechanism": "Ask about mechanisms/process/steps/causal chain.",
        "application": "Ask about uses, roles, or real-world effects.",
        "comparison": "Ask by contrasting with alternatives or baselines.",
        "cause": "Ask about causes, drivers, or background factors.",
        "effect": "Ask about outcomes, impacts, or verifiable results.",
        "condition": "Ask about conditions, scope, or constraints.",
        "example": "Ask for examples/evidence that can be located in the Body.",
        "quantitative": "Ask for numbers/metrics where a concrete value/range is answerable.",
        "temporal": "Ask along a timeline, stages, or milestones.",
    }
    default_by_round = {
        1: "Ask from varied angles: definition, causes, features, impacts, etc.",
        2: "Ask from deeper angles: mechanisms, processes, implementation steps, etc.",
        3: "Ask from application angles: use cases, effects, significance, etc.",
        4: "Ask from comparison angles: differences, pros/cons, relationships, etc.",
    }
    strategy = angle_strategies.get(view, default_by_round.get(attempt_round, default_by_round[1]))

    dedup_hint = ""
    if existing_questions:
        uniq_prev = [q.strip() for q in existing_questions if q and q.strip()]
        if uniq_prev:
            dedup_hint = "5) Avoid duplicates or near-paraphrases of existing questions: " + "; ".join(uniq_prev[:8])

    return f"""You are a question-generation expert. Create {number} high-quality questions based on the materials below.

Requirements:
1) The answer to each question MUST be directly recoverable from the Body (or by a single clear step).
2) Promote diversity - avoid semantic near-duplicates.
3) SOFT guidance only (do not fabricate): {kw_hint}
4) {strategy}
{dedup_hint}

Output strictly as a JSON array of strings:
["question 1","question 2","question 3"]

[Article Summary] {abstract}
[Chunk Summary] {chunk_summary}
[Keyword Hints] {kw_hint}
[Candidate Answer] {candidate_answer}

[Body]
{text}

Now produce {number} questions:"""


def build_answer_generation_prompt(question: str, chunk_text: str, candidate_answer: str) -> str:
    """Construct the multi-answer generation prompt."""

    return f"""Generate **3** accurate answers to the question **based strictly on the paragraph (Body)**. Answers must differ in wording but convey the same core meaning.

**Requirements:**
1. Answers must be fully grounded in the Body; no fabrication.
2. Each answer should use a different phrasing or sentence pattern (not just a single-word paraphrase), but keep the same core information.
3. Answers should directly address the question, concisely and clearly.
4. You may refer to the style of the candidate answer, but write with your own wording.
5. Prefer reusing exact numbers/units, variable names, table/figure refs if present (e.g., "29.7%-33.3%", "Variant A", "Table 3"). Avoid abstract paraphrases when concrete values exist.

**Output format:**
Return a strict JSON array: ["Answer 1", "Answer 2", "Answer 3"]

[Question] {question}

[Reference Answer] {candidate_answer}

[Body]
{chunk_text}

Now produce 3 answers:"""


def build_qa_pair_scoring_prompt(question: str, answer: str, chunk_text: str, abstract: str, keywords: List[str]) -> str:
    """Construct the scoring prompt used for QA evaluation."""

    keyword_str = ", ".join(keywords)
    return f"""Please **objectively score** the following QA pair on a 0-10 scale.

**Scoring rubric:**
1. **Answer Correctness (0-3):** Is the answer factually correct and does it answer the question?
   - 3: Fully correct; no factual errors
   - 2: Mostly correct; minor wording deviation
   - 1: Partially correct; noticeable inaccuracies
   - 0: Incorrect or off-topic

2. **Content Completeness (0-2):** Is the information sufficiently complete?
   - 2: Sufficient and comprehensive
   - 1: Mostly sufficient; minor omissions
   - 0: Insufficient; important parts missing

3. **QA Match (0-2):** Does the answer directly correspond to the question?
   - 2: Direct and precise match
   - 1: Generally matches with minor drift
   - 0: Poor match; deviates from the question

4. **Topical Relevance (0-3):** Is the QA pair strongly aligned with the article theme and keywords?
   - 3: Strongly aligned; core concepts accurate
   - 2: Aligned, but keyword match is moderate
   - 1: Somewhat related, but weak alignment
   - 0: Low relevance; misses core concepts

**Output format:**
```json
{{
  "total_score": 8.5,
  "dimension_scores": {{
    "Answer Correctness": 3,
    "Content Completeness": 2,
    "QA Match": 2,
    "Topical Relevance": 1.5
  }},
  "comment": "A brief comment (<=30 chars)",
  "reject_reason": "If total_score < 6, explain the main issue"
}}

Items to evaluate:
[Question] {question}
[Answer] {answer}

Reference materials:
[Body] {chunk_text}
[Article Theme] {abstract}
[Keywords] {keyword_str}

Scoring Requirements:
- Evaluate strictly based on the paragraph content (no external or fabricated information).
- Pay special attention to alignment with core keywords and topic.
- If total_score < 6, you must include a short explanation in reject_reason.
- Be fair and consistent (neither overly strict nor lenient).

Now output the evaluation as a valid JSON object:"""


def get_view_descriptions() -> dict:
    """
    Return concise descriptions for each QA view.
    Used for view embedding initialization in LinUCB FiLM.
    """
    return {
        "definition": "definition, scope, boundary, what it is",
        "mechanism": "mechanism, process, steps, causal chain",
        "application": "applications, use cases, roles, real-world effects",
        "comparison": "comparison, contrast, alternatives, pros and cons",
        "cause": "causes, drivers, background factors",
        "effect": "effects, outcomes, impacts, results",
        "condition": "conditions, constraints, applicability, scope",
        "example": "examples, evidence, instances from the text",
        "quantitative": "numbers, metrics, values, ranges, measurements",
        "temporal": "timeline, stages, milestones, chronological aspects",
    }
