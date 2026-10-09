# Bump when SYSTEM_PROMPT or context formatting changes; recorded in eval results.
PROMPT_VERSION = "v3"

SYSTEM_PROMPT = (
    "You are an accurate, concise assistant answering questions using ONLY "
    "the provided context chunks.\n"
    "\n"
    "STRICT GROUNDING RULES:\n"
    "1. Answer the user's question relying ONLY on the numbered context chunks below.\n"
    "2. Cite the supporting source chunk using [n] inline where 'n' is the chunk "
    "number (e.g., [1], [2]).\n"
    "3. If the context does not contain sufficient information to answer the question, "
    'state: "I don\'t have enough information in the provided context to answer '
    'this question."\n'
    "4. Do not invent facts, speculate, or use outside knowledge not supported by "
    "the context.\n"
    "5. Stay close to the wording of the context: never add field names, configuration "
    "examples, defaults, numbers, or steps that are not literally stated in the context."
)


def format_context_block(chunks: list) -> str:
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = getattr(chunk, "metadata", None) or {}
        doc_title = metadata.get("document_title") or "Document"
        page = metadata.get("page_start")
        page_str = f", Page {page}" if page else ""
        formatted.append(f"[{idx}] Source: {doc_title}{page_str}\n{chunk.text}\n")
    return "\n".join(formatted)


def format_memory_block(chunks: list) -> str:
    """Render chunks remembered from earlier turns as background-only context."""
    if not chunks:
        return ""
    lines = []
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        doc_title = metadata.get("document_title") or "Document"
        page = metadata.get("page_start")
        page_str = f", Page {page}" if page else ""
        lines.append(f"- Source: {doc_title}{page_str}\n  {chunk.text}")
    return (
        "EARLIER RETRIEVED CONTEXT (from previous turns of this conversation; "
        "background only — do NOT cite these with [n], cite only from the "
        "numbered CONTEXT CHUNKS):\n" + "\n".join(lines)
    )


def build_chat_messages(
    system_prompt: str,
    context_block: str,
    history: list,
    user_query: str,
    memory_block: str = "",
) -> list:
    system = f"{system_prompt}\n\nCONTEXT CHUNKS:\n{context_block}"
    if memory_block:
        system = f"{system}\n\n{memory_block}"
    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})
    return messages
