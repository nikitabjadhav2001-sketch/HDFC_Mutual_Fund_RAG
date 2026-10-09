from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    heading: str | None = None
    text: str
    page: int | None = None


class ParsedDoc(BaseModel):
    title: str
    sections: list[ParsedSection]


class Chunk(BaseModel):
    text: str
    ordinal: int
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    content_hash: str


class DocumentOut(BaseModel):
    id: str
    title: str
    format: str
    status: str
    error: str | None = None
    content_hash: str | None = None
    chunk_count: int = 0
    uploaded_at: str | None = None


class DocumentStatusOut(BaseModel):
    id: str
    status: str
    error: str | None = None
    chunk_count: int = 0


class UploadOut(BaseModel):
    document: DocumentOut
    created: bool = Field(description="false when an identical file was already ingested")


class ConversationOut(BaseModel):
    id: str
    title: str | None = None
    created_at: str | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict] | None = None
    created_at: str | None = None


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)
