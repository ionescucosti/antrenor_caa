"""Public API schemas. AnswerRequest is the internal, structured form of a
question; the Rokid client sends only `text` and/or `image` as a form."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QuestionType = Literal["multiple_choice", "open", "true_false", "single_choice"]
# aliases used by the Rokid client, normalised on input
_QUESTION_TYPE_ALIASES = {"multiple_select": "multiple_choice", "open_ended": "open", "single": "single_choice"}
AnswerStatus = Literal["answered", "insufficient_context", "invalid_question", "error"]


class Option(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    label: str = Field(min_length=1, max_length=5)
    text: str = Field(max_length=2000)


class ImagePayload(BaseModel):
    """Screenshot sent inline by the Rokid client (policy image_fallback)."""
    model_config = ConfigDict(extra="ignore")
    format: str = Field(default="jpeg", max_length=10)
    base64: str = Field(min_length=1, max_length=20_000_000)
    width: int | None = None
    height: int | None = None


class AnswerRequest(BaseModel):
    """Structured question as the Rokid client sends it (JSON). `source_mode`
    and other unknown fields are accepted and ignored."""
    model_config = ConfigDict(extra="ignore")
    request_id: str = Field(min_length=1, max_length=100)
    knowledge_base_id: str = Field(default="default", max_length=64)
    question: str = Field(default="", max_length=6000)
    question_type: QuestionType = "multiple_choice"
    options: list[Option] = Field(default_factory=list, max_length=12)
    raw_ocr_text: str | None = Field(default=None, max_length=12000)
    client_confidence: float | None = Field(default=None, ge=0, le=1)
    image: ImagePayload | None = None

    @field_validator("question_type", mode="before")
    @classmethod
    def _alias_type(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            return _QUESTION_TYPE_ALIASES.get(value, value)
        return value

    @field_validator("knowledge_base_id", mode="before")
    @classmethod
    def _default_kb(cls, value):
        return value if isinstance(value, str) and value.strip() else "default"

    @field_validator("options")
    @classmethod
    def _unique_labels(cls, options: list[Option]) -> list[Option]:
        labels = [o.label.upper() for o in options]
        if len(labels) != len(set(labels)):
            raise ValueError("Option labels must be unique.")
        return options

    def has_text(self) -> bool:
        return bool(self.question.strip() or (self.raw_ocr_text or "").strip())


class Answer(BaseModel):
    option_labels: list[str] = Field(default_factory=list)
    text: str = ""
    explanation: str = ""


class SourceRef(BaseModel):
    document_id: str
    document_name: str
    page: int | None = None
    section: str | None = None
    kind: Literal["document", "qa_dataset"] = "document"
    score: float | None = None


class AnswerResponse(BaseModel):
    request_id: str
    status: AnswerStatus
    answer: Answer | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    processing_time_ms: int = 0
    message: str = ""
    match_type: Literal["exact_qa", "semantic_qa", "rag", "general_knowledge", "none"] = "none"
    timings_ms: dict[str, float] = Field(default_factory=dict)
    transcribed_text: str | None = None   # set when the answer came from the uploaded image


class CompactAnswer(BaseModel):
    """Default reply of the question endpoints: only what the HUD must show."""
    label: str | None = None   # "A", or "A,C" for several correct options; null for open questions / no answer
    text: str | None = None    # text of the correct option(s) or the short answer
    error: str | None = None   # set only when there is no answer


# --- Administration ---------------------------------------------------------

class KnowledgeBaseIn(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class KnowledgeBaseOut(KnowledgeBaseIn):
    created_at: float
    document_count: int = 0
    qa_record_count: int = 0


class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    content_type: str
    size_bytes: int
    content_hash: str
    version: int
    status: str
    error: str
    warnings: list[Any]
    pages: int
    chunk_count: int
    embedding_provider: str
    created_at: float
    updated_at: float
    indexed_at: float | None


class QADatasetOut(BaseModel):
    id: str
    kb_id: str
    name: str
    filename: str
    format: str
    content_hash: str
    record_count: int
    invalid_count: int
    status: str
    error: str
    warnings: list[Any]
    embedding_provider: str
    created_at: float


class QARecordOut(BaseModel):
    id: str
    dataset_id: str
    question: str
    question_type: str
    options: list[Any]
    correct_labels: list[Any]
    correct_texts: list[Any]
    explanation: str
    source: dict[str, Any]


class JobOut(BaseModel):
    id: str
    kind: str
    target_id: str
    status: str
    message: str
    stages: dict[str, Any]
    started_at: float
    finished_at: float | None


class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=500)


class TokenIn(BaseModel):
    name: str = Field(default="", max_length=200)
    scope: Literal["device", "admin"] = "device"


class TokenOut(BaseModel):
    id: str
    name: str
    scope: str
    revoked: bool
    created_at: float
    last_used_at: float | None
    token: str | None = None  # only returned once, at creation
