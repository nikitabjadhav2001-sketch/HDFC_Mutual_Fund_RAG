import json
from pathlib import Path

import pytest

from eval import metrics as M
from eval.run_eval import (
    DEFAULT_DATASET,
    _aggregate,
    _gates,
    _thresholds,
    fixture_doc_id,
    load_dataset,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_corpus.json"


# --- dataset ------------------------------------------------------------------


def test_dataset_has_at_least_20_valid_entries():
    entries = load_dataset(DEFAULT_DATASET)
    assert len(entries) >= 20
    for entry in entries:
        assert entry["question"].strip()
        if entry["expect_refusal"]:
            assert entry["expected_doc_ids"] == []
        else:
            assert entry["expected_doc_ids"], entry["question"]
            assert entry["expected_keywords"], entry["question"]


def test_dataset_doc_ids_match_fixture_corpus_deterministically():
    entries = load_dataset(DEFAULT_DATASET)
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    known_ids = {fixture_doc_id(doc["title"]) for doc in corpus["documents"]}
    referenced = {doc_id for entry in entries for doc_id in entry["expected_doc_ids"]}
    assert referenced <= known_ids
    # deterministic: same title always maps to the same id
    assert fixture_doc_id("refunds") == fixture_doc_id("refunds")


def test_dataset_rejects_too_small_dataset(tmp_path):
    tiny = tmp_path / "tiny.jsonl"
    tiny.write_text(
        json.dumps(
            {
                "question": "q",
                "expected_doc_ids": ["x"],
                "expected_keywords": ["y"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=">= 20"):
        load_dataset(tiny)


# --- retrieval metrics --------------------------------------------------------


def test_hit_at_k_and_mrr():
    ranked = ["a", "b", "c", "d"]
    assert M.hit_at_k(ranked, ["c"], 1) == 0.0
    assert M.hit_at_k(ranked, ["c"], 3) == 1.0
    assert M.reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(ranked, ["missing"]) == 0.0
    assert M.hit_at_k([], ["c"], 5) == 0.0


def test_context_precision_and_recall():
    assert M.context_precision(["a", "a", "b", "c"], ["a"]) == 0.5
    assert M.context_precision([], ["a"]) == 0.0
    text = "Refunds arrive within 30 days of delivery."
    assert M.context_recall(text, ["30 days", "missing"]) == 0.5
    assert M.context_recall(text, []) == 1.0


# --- generation metrics -------------------------------------------------------


CONTEXT = (
    "Our refund policy allows customers to return any purchase within 30 days "
    "of delivery. Refunds are issued to the original payment method within five "
    "business days of the return arriving at our warehouse."
)


def test_faithfulness_grounded_answer_scores_high():
    answer = "Refunds are issued within 30 days of delivery [1]."
    assert M.faithfulness(answer, CONTEXT) == 1.0


def test_faithfulness_penalizes_hallucination():
    answer = "Refunds are processed instantly by our teleportation portal [1]."
    assert M.faithfulness(answer, CONTEXT) < 0.5


def test_faithfulness_empty_answer_scores_zero():
    assert M.faithfulness("", CONTEXT) == 0.0


def test_faithfulness_ignores_citation_markers_and_short_clauses():
    answer = "Yes. Within 30 days [1]."
    assert M.faithfulness(answer, CONTEXT) == 1.0


def test_answer_relevancy_reflects_question_words():
    question = "What is the refund window for damaged items?"
    assert M.answer_relevancy(question, "The refund window is 30 days from delivery.") > 0.3
    assert M.answer_relevancy(question, "") == 0.0


def test_token_f1_reference_scoring():
    assert M.token_f1("red fox jumps", "the red fox jumps high") == pytest.approx(0.857, abs=0.01)
    assert M.token_f1("alpha beta", "gamma delta") == 0.0


def test_citation_correctness():
    assert M.citation_correctness("Answer [1] and [2].", 3) == 1.0
    assert M.citation_correctness("Answer [9].", 3) == 0.0
    assert M.citation_correctness("No citations here.", 3) is None


def test_is_refusal():
    assert M.is_refusal("I don't have enough information in the provided context.")
    assert not M.is_refusal("The refund window is 30 days.")


# --- aggregation + gates ------------------------------------------------------


def _row(question="q", expect_refusal=False, hit5=1.0, faithfulness=1.0, refused=False):
    row = {
        "question": question,
        "expect_refusal": expect_refusal,
        "retrieval": {
            "hit@1": hit5,
            "hit@3": hit5,
            "hit@5": hit5,
            "mrr": hit5,
            "context_precision": hit5,
            "context_recall": hit5,
            "latency_ms": 10.0,
        },
        "generation": None
        if expect_refusal and refused
        else {
            "refused": refused,
            "faithfulness": faithfulness,
            "answer_relevancy": 0.5,
            "token_f1": 0.5,
            "citation_correctness": 1.0,
            "first_token_ms": 100.0,
        },
    }
    if expect_refusal:
        for key in ("hit@1", "hit@3", "hit@5", "mrr", "context_precision", "context_recall"):
            row["retrieval"].pop(key)
        if refused:
            row["generation"] = {
                "refused": True,
                "faithfulness": 1.0,
                "answer_relevancy": None,
                "token_f1": None,
                "citation_correctness": None,
                "first_token_ms": 100.0,
            }
    return row


def test_aggregate_and_gates_pass_on_good_metrics():
    rows = [_row() for _ in range(10)]
    rows.append(_row(question="unanswerable", expect_refusal=True, refused=True))
    summary = _aggregate(rows, skip_generation=False)
    assert summary["retrieval"]["hit_at_5"] == 1.0
    assert summary["generation"]["faithfulness"] == 1.0
    assert summary["generation"]["refusal_accuracy"] == 1.0

    checks = _gates(summary, _thresholds())
    assert all(check["passed"] for check in checks)


def test_gates_fail_when_hit5_or_faithfulness_below_threshold():
    rows = [_row(hit5=0.0, faithfulness=0.2) for _ in range(10)]
    summary = _aggregate(rows, skip_generation=False)
    checks = _gates(summary, {"hit_at_5": 0.8, "faithfulness": 0.9})
    assert {check["metric"]: check["passed"] for check in checks} == {
        "hit_at_5": False,
        "faithfulness": False,
    }


def test_gates_skip_faithfulness_when_generation_skipped():
    rows = [_row() for _ in range(5)]
    summary = _aggregate(rows, skip_generation=True)
    assert summary["generation"] is None
    checks = _gates(summary, _thresholds())
    assert [check["metric"] for check in checks] == ["hit_at_5"]


def test_refusal_on_answerable_question_is_penalized():
    rows = [_row(refused=True, faithfulness=0.0) for _ in range(5)]
    summary = _aggregate(rows, skip_generation=False)
    assert summary["generation"]["faithfulness"] == 0.0
    assert summary["generation"]["refusal_accuracy"] == 0.0
