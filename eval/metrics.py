"""Deterministic RAGAS-style eval metrics (no LLM judge, no network).

Generation metrics mirror the RAGAS names from the implementation plan
(faithfulness, answer relevancy, citation correctness) but compute them with
lexical claim-support checks so `make eval` is fast, free and reproducible.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CITATION_RE = re.compile(r"\[(\d+)\]")


def _stem(token: str) -> str:
    """Conservative suffix stripping applied to both sides of every comparison."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ly"):
        return token[:-2]
    return token

STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against between into
    through during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some such no
    nor not only own same so than too very can will just do does did doing would
    should could of is are was were be been being have has had having i me my we our
    you your he she it its they them their what which who whom this that these those
    am as
    """.split()
)

DEFAULT_REFUSAL_PHRASE = "don't have enough information"


def content_words(text: str) -> list[str]:
    """Lowercased, stopword-free, lightly stemmed tokens (numbers kept)."""
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def is_refusal(answer: str, refusal_phrase: str = DEFAULT_REFUSAL_PHRASE) -> bool:
    return refusal_phrase in answer.lower()


# --- retrieval metrics --------------------------------------------------------


def hit_at_k(ranked_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """1.0 if any expected document appears in the top-k ranked chunks."""
    if not expected_doc_ids:
        raise ValueError("hit_at_k requires at least one expected doc id")
    return 1.0 if any(doc_id in expected_doc_ids for doc_id in ranked_doc_ids[:k]) else 0.0


def reciprocal_rank(ranked_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0


def context_precision(ranked_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of retrieved context chunks that come from expected documents."""
    if not ranked_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids if doc_id in expected_doc_ids)
    return hits / len(ranked_doc_ids)


def context_recall(context_text: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the assembled context."""
    if not expected_keywords:
        return 1.0
    haystack = context_text.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in haystack)
    return hits / len(expected_keywords)


# --- generation metrics -------------------------------------------------------


def _split_claims(answer: str) -> list[str]:
    cleaned = CITATION_RE.sub("", answer)
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def faithfulness(answer: str, context_text: str) -> float:
    """Fraction of answer claims supported by the retrieved context.

    A claim is supported when >= 50% of its content words occur in the context.
    Empty answers score 0; claims with fewer than two content words are ignored.
    """
    if not content_words(answer):
        return 0.0
    context_vocab = set(content_words(context_text))
    evaluated = 0
    supported = 0
    for claim in _split_claims(answer):
        words = content_words(claim)
        if len(words) < 2:
            continue
        evaluated += 1
        overlap = sum(1 for word in words if word in context_vocab)
        if overlap / len(words) >= 0.5:
            supported += 1
    if evaluated == 0:
        return 1.0
    return supported / evaluated


def answer_relevancy(question: str, answer: str) -> float:
    """Fraction of question content words reflected in the answer."""
    question_words = content_words(question)
    if not question_words:
        return 1.0
    answer_vocab = set(content_words(answer))
    if not answer_vocab:
        return 0.0
    hits = sum(1 for word in question_words if word in answer_vocab)
    return hits / len(question_words)


def token_f1(reference: str, prediction: str) -> float:
    """Standard QA token-level F1 between a reference answer and a prediction."""
    ref_tokens = content_words(reference)
    pred_tokens = content_words(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_correctness(answer: str, context_size: int) -> float | None:
    """Validity of [n] citation markers against the assembled context size.

    Returns None when the answer carries no citations (not applicable).
    A grounded non-refusal answer is expected to cite, so callers treat
    "no markers" as a miss rather than skipping the metric.
    """
    markers = [int(idx) for idx in CITATION_RE.findall(answer)]
    if not markers:
        return None
    valid = sum(1 for idx in markers if 1 <= idx <= max(context_size, 1))
    return valid / len(markers)
