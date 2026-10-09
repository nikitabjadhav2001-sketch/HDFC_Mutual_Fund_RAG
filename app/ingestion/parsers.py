import io
import re
from pathlib import PurePosixPath

from app.schemas import ParsedDoc, ParsedSection

SUPPORTED_FORMATS = {"txt", "md", "markdown", "pdf", "docx"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HEADING_LEVEL = re.compile(r"^Heading\s+(\d+)")


class ParseError(Exception):
    """Raised when a document cannot be parsed into text."""


def detect_format(filename: str) -> str:
    ext = PurePosixPath(filename.replace("\\", "/")).suffix.lstrip(".").lower()
    if ext == "markdown":
        return "md"
    if ext not in SUPPORTED_FORMATS:
        raise ParseError(f"unsupported file format: {ext or 'unknown'}")
    return ext


def parse(file_bytes: bytes, format: str, *, filename: str | None = None) -> ParsedDoc:
    fmt = format.lower()
    if fmt == "markdown":
        fmt = "md"
    if fmt not in SUPPORTED_FORMATS:
        raise ParseError(f"unsupported file format: {fmt}")
    if not file_bytes:
        raise ParseError("empty file")

    if fmt == "txt":
        doc = _parse_txt(file_bytes, filename)
    elif fmt == "md":
        doc = _parse_md(file_bytes, filename)
    elif fmt == "pdf":
        doc = _parse_pdf(file_bytes, filename)
    else:
        doc = _parse_docx(file_bytes, filename)

    if not any(section.text.strip() for section in doc.sections):
        raise ParseError("no extractable text found")
    return doc


def _title(filename: str | None, fallback: str = "untitled") -> str:
    if not filename:
        return fallback
    stem = PurePosixPath(filename.replace("\\", "/")).stem
    return stem or fallback


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _parse_txt(file_bytes: bytes, filename: str | None) -> ParsedDoc:
    return ParsedDoc(title=_title(filename), sections=[ParsedSection(text=_decode(file_bytes))])


def _parse_md(file_bytes: bytes, filename: str | None) -> ParsedDoc:
    lines = _decode(file_bytes).splitlines()
    title = _title(filename)
    sections: list[ParsedSection] = []
    heading: str | None = None
    buf: list[str] = []
    saw_h1 = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            sections.append(ParsedSection(heading=heading, text=text))

    for line in lines:
        match = _MD_HEADING.match(line)
        if match:
            flush()
            level, text = len(match.group(1)), match.group(2)
            heading = text if level > 1 else None
            if level == 1 and not saw_h1:
                saw_h1 = True
                title = text
        else:
            buf.append(line)
    flush()
    return ParsedDoc(title=title, sections=sections)


def _parse_pdf(file_bytes: bytes, filename: str | None) -> ParsedDoc:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            reader.decrypt("")
        sections = [
            ParsedSection(heading=None, text=(page.extract_text() or ""), page=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"invalid PDF: {exc}") from exc

    meta_title = None
    try:
        meta_title = reader.metadata.title if reader.metadata else None
    except Exception:
        meta_title = None
    title = (meta_title or "").strip() or _title(filename)
    return ParsedDoc(title=title, sections=sections)


def _parse_docx(file_bytes: bytes, filename: str | None) -> ParsedDoc:
    from docx import Document as DocxDocument

    try:
        document = DocxDocument(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ParseError(f"invalid DOCX: {exc}") from exc

    title = _title(filename)
    if document.core_properties.title:
        title = document.core_properties.title.strip() or title

    heading_stack: list[tuple[int, str]] = []
    sections: list[ParsedSection] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n\n".join(buf).strip()
        buf.clear()
        if text:
            path = " > ".join(text for _, text in heading_stack) or None
            sections.append(ParsedSection(heading=path, text=text))

    for paragraph in document.paragraphs:
        style = paragraph.style.name if paragraph.style is not None else ""
        level_match = _HEADING_LEVEL.match(style)
        if level_match:
            flush()
            level = int(level_match.group(1))
            text = paragraph.text.strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if text:
                heading_stack.append((level, text))
            continue
        if paragraph.text.strip():
            buf.append(paragraph.text)
    flush()

    return ParsedDoc(title=title, sections=sections)
