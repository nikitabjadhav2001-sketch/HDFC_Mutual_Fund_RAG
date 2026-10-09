import pytest

from app.ingestion.chunker import chunk_document, estimate_tokens
from app.schemas import ParsedDoc, ParsedSection

MAX_TOKENS = 200
OVERLAP_TOKENS = 40


def _long_section(word: str = "alpha", count: int = 400) -> ParsedSection:
    words = [f"{word}{i}" for i in range(count)]
    paragraphs = [" ".join(words[i : i + 25]) for i in range(0, count, 25)]
    return ParsedSection(heading="Intro", text="\n\n".join(paragraphs), page=1)


def _chunks(doc: ParsedDoc, **kwargs) -> list:
    return chunk_document(doc, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, **kwargs)


def test_chunks_respect_token_budget() -> None:
    doc = ParsedDoc(title="t", sections=[_long_section()])
    chunks = _chunks(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= MAX_TOKENS


def test_overlap_between_consecutive_chunks() -> None:
    chunks = _chunks(ParsedDoc(title="t", sections=[_long_section()]))
    for previous, following in zip(chunks, chunks[1:], strict=False):
        prev_words = previous.text.split()
        next_words = following.text.split()
        overlap = min(len(prev_words), len(next_words))
        while overlap > 0 and prev_words[-overlap:] != next_words[:overlap]:
            overlap -= 1
        assert overlap >= 5, "consecutive chunks must share an overlap"


def test_section_boundaries_and_paths_are_preserved() -> None:
    doc = ParsedDoc(
        title="t",
        sections=[
            ParsedSection(heading="One", text=" ".join(f"w{i}" for i in range(200))),
            ParsedSection(heading="Two", text=" ".join(f"z{i}" for i in range(200))),
        ],
    )
    chunks = _chunks(doc)
    seen = {chunk.section_path for chunk in chunks}
    assert seen == {"One", "Two"}
    for chunk in chunks:
        if chunk.section_path == "One":
            assert "z0" not in chunk.text
        else:
            assert "w0" not in chunk.text


def test_words_are_preserved_in_order() -> None:
    section = _long_section(count=300)
    chunks = _chunks(ParsedDoc(title="t", sections=[section]))
    remaining = section.text.split()
    position = 0
    for chunk in chunks:
        for word in chunk.text.split():
            if position < len(remaining) and word == remaining[position]:
                position += 1
    assert position == len(remaining)


def test_ordinals_are_sequential_and_hashes_distinct() -> None:
    doc = ParsedDoc(
        title="t",
        sections=[
            ParsedSection(
                heading="A", text=" ".join(f"first{i}" for i in range(120))
            ),
            ParsedSection(
                heading="B", text=" ".join(f"second{i}" for i in range(120))
            ),
        ],
    )
    chunks = _chunks(doc)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    hashes = [chunk.content_hash for chunk in chunks]
    assert len(set(hashes)) == len(hashes)
    assert all(len(value) == 64 for value in hashes)


def test_content_hash_is_whitespace_insensitive() -> None:
    doc_a = ParsedDoc(title="t", sections=[ParsedSection(text="hello   world\n\nagain")])
    doc_b = ParsedDoc(title="t", sections=[ParsedSection(text="hello world again")])
    # identical normalized text -> identical hash when chunked the same way
    hash_a = _chunks(doc_a)[0].content_hash
    chunks_b = _chunks(doc_b)
    assert hash_a == chunks_b[0].content_hash


def test_empty_sections_are_skipped() -> None:
    doc = ParsedDoc(
        title="t",
        sections=[ParsedSection(text="   "), ParsedSection(heading="Keep", text="body text")],
    )
    chunks = _chunks(doc)
    assert len(chunks) == 1
    assert chunks[0].section_path == "Keep"


def test_page_numbers_flow_into_chunks() -> None:
    doc = ParsedDoc(
        title="t",
        sections=[ParsedSection(text="page two content", page=2)],
    )
    chunk = _chunks(doc)[0]
    assert chunk.page_start == 2
    assert chunk.page_end == 2


def test_invalid_parameters_raise() -> None:
    doc = ParsedDoc(title="t", sections=[ParsedSection(text="x")])
    with pytest.raises(ValueError):
        chunk_document(doc, max_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        chunk_document(doc, max_tokens=60, overlap_tokens=60)
    with pytest.raises(ValueError):
        chunk_document(doc, max_tokens=60, overlap_tokens=-1)
