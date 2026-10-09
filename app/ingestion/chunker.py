import hashlib
import math
import re

from app.config import get_settings
from app.schemas import Chunk, ParsedDoc


def estimate_tokens(text: str) -> int:
    """Approximate token count (chars/4) — avoids a tokenizer dependency."""
    return max(1, math.ceil(len(text) / 4))


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chunk_document(
    doc: ParsedDoc,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
    overlap_tokens = (
        overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens
    )
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    chunks: list[Chunk] = []
    ordinal = 0
    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue
        atoms = _atoms(text, max_tokens)
        carry = ""
        buf: list[str] = []

        def compose(parts: list[str]) -> str:
            body = "\n\n".join(parts)
            return f"{carry}\n{body}" if carry else body

        def emit() -> None:
            nonlocal ordinal, carry, buf
            if not buf:
                return
            body = "\n\n".join(buf)
            chunk_text = compose(buf)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    ordinal=ordinal,
                    section_path=section.heading,
                    page_start=section.page,
                    page_end=section.page,
                    token_count=estimate_tokens(chunk_text),
                    content_hash=_content_hash(chunk_text),
                )
            )
            ordinal += 1
            carry = _tail(body, overlap_tokens)
            buf = []

        for atom in atoms:
            if buf and estimate_tokens(compose([*buf, atom])) > max_tokens:
                emit()
            while not buf and carry and estimate_tokens(compose([atom])) > max_tokens:
                carry = " ".join(carry.split()[1:])
            buf.append(atom)
        emit()
    return chunks


def _atoms(text: str, max_tokens: int) -> list[str]:
    """Split section text into paragraphs; split oversized paragraphs by words."""
    atoms: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if estimate_tokens(paragraph) <= max_tokens:
            atoms.append(paragraph)
            continue
        line: list[str] = []
        for word in paragraph.split():
            trial = " ".join([*line, word])
            if line and estimate_tokens(trial) > max_tokens:
                atoms.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            atoms.append(" ".join(line))
    return atoms


def _tail(body: str, overlap_tokens: int) -> str:
    """Last ~overlap_tokens worth of words from an emitted chunk body."""
    if overlap_tokens <= 0:
        return ""
    words = body.split()
    tail: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word, *tail])
        if tail and estimate_tokens(candidate) > overlap_tokens:
            break
        tail.insert(0, word)
    return " ".join(tail)
