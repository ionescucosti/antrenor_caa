"""Relational metadata (SQLite by default). ChromaDB only holds vectors."""
from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


class Base(DeclarativeBase):
    pass


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), ForeignKey("knowledge_bases.id"), index=True)
    filename: Mapped[str] = mapped_column(String(400), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending|processing|indexed|failed
    error: Mapped[str] = mapped_column(Text, default="")
    warnings: Mapped[list[Any]] = mapped_column(JSON, default=list)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    stored_path: Mapped[str] = mapped_column(String(600), default="")
    embedding_provider: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)
    updated_at: Mapped[float] = mapped_column(Float, default=now, onupdate=now)
    indexed_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class QADataset(Base):
    __tablename__ = "qa_datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), ForeignKey("knowledge_bases.id"), index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    filename: Mapped[str] = mapped_column(String(400), default="")
    format: Mapped[str] = mapped_column(String(30), default="json")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    invalid_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    warnings: Mapped[list[Any]] = mapped_column(JSON, default=list)
    stored_path: Mapped[str] = mapped_column(String(600), default="")
    embedding_provider: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)
    records: Mapped[list["QARecord"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class QARecord(Base):
    __tablename__ = "qa_records"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), ForeignKey("qa_datasets.id"), index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_question: Mapped[str] = mapped_column(Text, nullable=False)
    question_hash: Mapped[str] = mapped_column(String(64), index=True)
    question_type: Mapped[str] = mapped_column(String(30), default="multiple_choice")
    options: Mapped[list[Any]] = mapped_column(JSON, default=list)          # [{"label","text"}]
    correct_labels: Mapped[list[Any]] = mapped_column(JSON, default=list)   # original labels
    correct_texts: Mapped[list[Any]] = mapped_column(JSON, default=list)    # option texts (or free answer)
    explanation: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)      # file/page/number/line
    created_at: Mapped[float] = mapped_column(Float, default=now)
    dataset: Mapped[QADataset] = relationship(back_populates="records")


Index("qa_records_kb_hash", QARecord.kb_id, QARecord.question_hash)


class AnswerCandidate(Base):
    """An answer given from documentation / web search (not from a verified
    set), kept for review: confirming it creates a verified QARecord."""
    __tablename__ = "answer_candidates"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_question: Mapped[str] = mapped_column(Text, default="")
    question_hash: Mapped[str] = mapped_column(String(64), index=True)
    options: Mapped[list[Any]] = mapped_column(JSON, default=list)          # [{"label","text"}]
    option_labels: Mapped[list[Any]] = mapped_column(JSON, default=list)    # proposed answer
    answer_text: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    agreement: Mapped[str] = mapped_column(String(30), default="none")      # consensus|docs_only|web_only|disagreement
    match_type: Mapped[str] = mapped_column(String(30), default="rag")
    docs_labels: Mapped[list[Any]] = mapped_column(JSON, default=list)
    web_labels: Mapped[list[Any]] = mapped_column(JSON, default=list)
    sources: Mapped[list[Any]] = mapped_column(JSON, default=list)
    request_id: Mapped[str] = mapped_column(String(100), default="")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending|confirmed|rejected
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)
    updated_at: Mapped[float] = mapped_column(Float, default=now, onupdate=now)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))          # document|qa_dataset
    target_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")  # queued|running|done|failed
    message: Mapped[str] = mapped_column(Text, default="")
    stages: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[float] = mapped_column(Float, default=now)
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class ApiToken(Base):
    __tablename__ = "api_tokens"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scope: Mapped[str] = mapped_column(String(20), default="device")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[float] = mapped_column(Float, default=now)
    last_used_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
