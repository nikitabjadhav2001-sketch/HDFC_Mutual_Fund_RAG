"""Evaluation harness: golden-set retrieval + generation metrics with quality gates.

Usage:
    python -m eval.run_eval                  # full pipeline, writes eval/results/
    python -m eval.run_eval --skip-generation
    make eval                                # same thing, via the Makefile

Pipeline per question: seed fixture corpus (idempotent) → hybrid retrieve →
metric scoring → stream the grounded answer → metric scoring. Results are
written to eval/results/{timestamp}.json (and latest.json) with the prompt
version and git sha recorded; the process exits non-zero when a gated metric
(faithfulness, hit@5) falls below its threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select

from app.config import get_settings
from app.core.embedding import build_embedder
from app.core.sparse import tokenize
from app.db import SessionLocal
from app.models.tables import Chunk as ChunkRow
from app.models.tables import Document
from app.rag.orchestrator import GenerationOrchestrator
from app.rag.prompts import PROMPT_VERSION
from eval import metrics as M

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "retrieval_corpus.json"
RESULTS_DIR = ROOT / "eval" / "results"

DOC_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-eval-fixture-corpus")

DEFAULT_THRESHOLDS = {"faithfulness": 0.9, "hit_at_5": 0.8}


def _thresholds() -> dict[str, float]:
    return {
        "faithfulness": float(
            os.environ.get("EVAL_FAITHFULNESS_THRESHOLD", DEFAULT_THRESHOLDS["faithfulness"])
        ),
        "hit_at_5": float(
            os.environ.get("EVAL_HIT_AT_5_THRESHOLD", DEFAULT_THRESHOLDS["hit_at_5"])
        ),
    }


def fixture_doc_id(title: str) -> str:
    return str(uuid.uuid5(DOC_ID_NAMESPACE, title))


def load_dataset(path: Path) -> list[dict]:
    entries = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        missing = {"question", "expected_doc_ids", "expected_keywords"} - entry.keys()
        if missing:
            raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
        entry.setdefault("expect_refusal", False)
        entries.append(entry)
    if len(entries) < 20:
        raise ValueError(f"{path} has {len(entries)} entries; the golden set needs >= 20")
    return entries


def git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def db_reachable() -> bool:
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(get_settings().database_url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


def seed_fixture_corpus() -> dict[str, str]:
    """Idempotently index the fixture corpus with deterministic document ids.

    Fixture documents are tagged (`eval_fixture`) and re-seeded only when the
    configured embedder changes, so user documents are never touched.
    """
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    embedder = build_embedder()
    tags = {"eval_fixture": True, "embedder": embedder.model}

    title_to_id: dict[str, str] = {}
    with SessionLocal() as db:
        for doc in corpus["documents"]:
            doc_id = uuid.UUID(fixture_doc_id(doc["title"]))
            title_to_id[doc["title"]] = str(doc_id)
            chunks = doc["chunks"]

            existing = db.get(Document, doc_id)
            chunk_count = db.execute(
                select(func.count()).select_from(ChunkRow).where(ChunkRow.doc_id == doc_id)
            ).scalar_one()
            fresh = existing is not None and existing.tags == tags and chunk_count == len(chunks)
            if fresh:
                continue
            # Embed only stale documents so a fresh fixture corpus costs no API calls.
            vectors = embedder.embed_texts(chunks)
            db.execute(delete(ChunkRow).where(ChunkRow.doc_id == doc_id))
            if existing is None:
                db.add(
                    Document(
                        id=doc_id,
                        title=doc["title"],
                        format="md",
                        content_hash=hashlib.sha256(doc["title"].encode()).hexdigest(),
                        status="done",
                        tags=tags,
                    )
                )
            else:
                existing.tags = tags
                existing.status = "done"
                existing.error = None
            db.flush()  # documents row must exist before its chunks (FK)
            for ordinal, text in enumerate(chunks):
                db.add(
                    ChunkRow(
                        doc_id=doc_id,
                        text=text,
                        embedding=vectors[ordinal],
                        token_count=len(tokenize(text)),
                        ordinal=ordinal,
                        content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    )
                )
        db.commit()
    return title_to_id


async def _generate(orchestrator: GenerationOrchestrator, question: str, result) -> dict:
    answer_parts: list[str] = []
    sources: list = []
    first_token_ms: float | None = None
    started = time.perf_counter()
    async for event in orchestrator.stream_response(
        question, k=5, retrieval=result, use_cache=False
    ):
        if event["type"] == "token":
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            answer_parts.append(event.get("content", ""))
        elif event["type"] == "sources":
            sources = event.get("sources", [])
    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "first_token_ms": round(first_token_ms or 0.0, 1),
    }


def _score_generation(entry: dict, answer: str, context_text: str, context_size: int) -> dict:
    refused = M.is_refusal(answer)
    expect_refusal = entry["expect_refusal"]

    if expect_refusal:
        faithfulness = 1.0 if refused else M.faithfulness(answer, context_text)
    elif refused:
        faithfulness = 0.0  # blanket refusal on an answerable question is a failure
    else:
        faithfulness = M.faithfulness(answer, context_text)

    if expect_refusal:
        relevancy = None
        token_f1 = None
    elif refused:
        relevancy, token_f1 = 0.0, 0.0
    else:
        relevancy = M.answer_relevancy(entry["question"], answer)
        token_f1 = (
            M.token_f1(entry["acceptable_answer"], answer)
            if entry.get("acceptable_answer")
            else None
        )

    citations = None
    if not refused:
        citations = M.citation_correctness(answer, context_size)
        if citations is None:  # grounded answer without citations counts as a miss
            citations = 0.0

    return {
        "refused": refused,
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": None if relevancy is None else round(relevancy, 4),
        "token_f1": None if token_f1 is None else round(token_f1, 4),
        "citation_correctness": None if citations is None else round(citations, 4),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 1)


def _aggregate(rows: list[dict], skip_generation: bool) -> dict:
    answerable = [row for row in rows if not row["expect_refusal"]]

    retrieval = {
        "hit_at_1": _mean([row["retrieval"]["hit@1"] for row in answerable]),
        "hit_at_3": _mean([row["retrieval"]["hit@3"] for row in answerable]),
        "hit_at_5": _mean([row["retrieval"]["hit@5"] for row in answerable]),
        "mrr": _mean([row["retrieval"]["mrr"] for row in answerable]),
        "context_precision": _mean([row["retrieval"]["context_precision"] for row in answerable]),
        "context_recall": _mean([row["retrieval"]["context_recall"] for row in answerable]),
        "latency_ms_p95": _p95([row["retrieval"]["latency_ms"] for row in rows]),
    }

    if skip_generation:
        return {"retrieval": retrieval, "generation": None}

    scored = [row["generation"] for row in rows if row["generation"]]
    generation = {
        "faithfulness": _mean([s["faithfulness"] for s in scored]),
        "answer_relevancy": _mean(
            [s["answer_relevancy"] for s in scored if s["answer_relevancy"] is not None]
        ),
        "token_f1": _mean([s["token_f1"] for s in scored if s["token_f1"] is not None]),
        "citation_correctness": _mean(
            [s["citation_correctness"] for s in scored if s["citation_correctness"] is not None]
        ),
        "refusal_accuracy": _mean(
            [
                1.0 if (row["generation"]["refused"] == row["expect_refusal"]) else 0.0
                for row in rows
                if row["generation"]
            ]
        ),
        "first_token_ms_p95": _p95(
            [row["generation"]["first_token_ms"] for row in rows if row["generation"]]
        ),
    }
    return {"retrieval": retrieval, "generation": generation}


def _gates(summary: dict, thresholds: dict[str, float]) -> list[dict]:
    checks = []
    hit_at_5 = summary["retrieval"]["hit_at_5"]
    checks.append(
        {
            "metric": "hit_at_5",
            "value": hit_at_5,
            "threshold": thresholds["hit_at_5"],
            "passed": hit_at_5 is not None and hit_at_5 >= thresholds["hit_at_5"],
        }
    )
    if summary["generation"] is not None:
        faithfulness = summary["generation"]["faithfulness"]
        checks.append(
            {
                "metric": "faithfulness",
                "value": faithfulness,
                "threshold": thresholds["faithfulness"],
                "passed": faithfulness is not None and faithfulness >= thresholds["faithfulness"],
            }
        )
    return checks


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not db_reachable():
        print(
            "eval: PostgreSQL is not reachable — start it (docker compose up postgres)",
            file=sys.stderr,
        )
        return 2

    dataset_path = Path(args.dataset)
    entries = load_dataset(dataset_path)
    if args.limit:
        entries = entries[: args.limit]

    seed_fixture_corpus()
    print(f"eval: {len(entries)} entries from {dataset_path.name} (corpus seeded)")

    orchestrator = GenerationOrchestrator()
    rows: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        question = entry["question"]
        started = time.perf_counter()
        try:
            result = await orchestrator.retriever.retrieve(question, k=settings.retrieval_top_k)
        except Exception as exc:  # noqa: BLE001 - embedding/DB outage should fail the run
            print(f"eval: retrieval failed for entry {index}: {exc}", file=sys.stderr)
            return 2
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

        ranked_doc_ids = [chunk.doc_id for chunk in result.chunks]
        context_text = "\n".join(chunk.text for chunk in result.chunks)
        expected = entry["expected_doc_ids"]

        row: dict = {
            "question": question,
            "expect_refusal": entry["expect_refusal"],
            "retrieval": {
                "is_sufficient": result.is_sufficient,
                "top_score": round(result.top_score, 4),
                "context_size": len(result.chunks),
                "latency_ms": retrieval_ms,
            },
            "generation": None,
        }
        if expected:
            row["retrieval"].update(
                {
                    "hit@1": M.hit_at_k(ranked_doc_ids, expected, 1),
                    "hit@3": M.hit_at_k(ranked_doc_ids, expected, 3),
                    "hit@5": M.hit_at_k(ranked_doc_ids, expected, 5),
                    "mrr": round(M.reciprocal_rank(ranked_doc_ids, expected), 4),
                    "context_precision": round(M.context_precision(ranked_doc_ids, expected), 4),
                    "context_recall": round(
                        M.context_recall(context_text, entry["expected_keywords"]), 4
                    ),
                }
            )

        if not args.skip_generation:
            generated = await _generate(orchestrator, question, result)
            generation = _score_generation(
                entry, generated["answer"], context_text, len(result.chunks)
            )
            generation.update(
                {
                    "answer": generated["answer"],
                    "source_count": len(generated["sources"]),
                    "first_token_ms": generated["first_token_ms"],
                }
            )
            row["generation"] = generation

        rows.append(row)
        status = "REFUSED" if row["generation"] and row["generation"]["refused"] else "ok"
        print(f"  [{index:02d}/{len(entries)}] {status:7s} {retrieval_ms:7.1f}ms  {question}")

    thresholds = _thresholds()
    summary = _aggregate(rows, args.skip_generation)
    checks = _gates(summary, thresholds)
    passed = all(check["passed"] for check in checks)

    dataset_display = (
        str(dataset_path.relative_to(ROOT))
        if dataset_path.is_relative_to(ROOT)
        else str(dataset_path)
    )
    results = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset_display,
        "entries": len(rows),
        "git_sha": git_sha(),
        "prompt_version": PROMPT_VERSION,
        "llm_model": None if args.skip_generation else settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": build_embedder().model,
        "thresholds": thresholds,
        "summary": summary,
        "gates": checks,
        "pass": passed,
        "per_entry": rows,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_path = results_dir / f"{stamp}.json"
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    result_path.write_text(payload, encoding="utf-8")
    (results_dir / "latest.json").write_text(payload, encoding="utf-8")

    print("\nsummary:")
    for name, value in summary["retrieval"].items():
        print(f"  retrieval.{name:22s} = {value}")
    if summary["generation"]:
        for name, value in summary["generation"].items():
            print(f"  generation.{name:20s} = {value}")
    for check in checks:
        verdict = "PASS" if check["passed"] else "FAIL"
        print(f"  gate {check['metric']:12s} {check['value']} >= {check['threshold']} -> {verdict}")
    print(f"\nresults: {result_path.relative_to(ROOT)} (latest.json updated)")
    print("eval PASSED" if passed else "eval FAILED")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="retrieval metrics only (no LLM calls)",
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N entries")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
