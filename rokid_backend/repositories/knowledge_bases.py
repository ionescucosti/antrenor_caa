from __future__ import annotations

from sqlalchemy import func, select

from rokid_backend.models import Document, KnowledgeBase, QADataset, QARecord


class KnowledgeBaseError(ValueError):
    pass


class KnowledgeBaseService:
    def __init__(self, db, store) -> None:
        self.db = db
        self.store = store

    def ensure_default(self) -> None:
        with self.db.session() as session:
            if session.get(KnowledgeBase, "default") is None:
                session.add(KnowledgeBase(id="default", name="Default", description="Baza de cunostinte implicita."))

    def exists(self, kb_id: str) -> bool:
        with self.db.session() as session:
            return session.get(KnowledgeBase, kb_id) is not None

    def list(self) -> list[dict]:
        with self.db.session() as session:
            out = []
            for kb in session.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at)).scalars():
                docs = session.execute(select(func.count()).select_from(Document).where(Document.kb_id == kb.id)).scalar() or 0
                recs = session.execute(select(func.count()).select_from(QARecord).where(QARecord.kb_id == kb.id)).scalar() or 0
                out.append({"id": kb.id, "name": kb.name, "description": kb.description, "created_at": kb.created_at,
                            "document_count": docs, "qa_record_count": recs})
            return out

    def create(self, kb_id: str, name: str, description: str = "") -> dict:
        with self.db.session() as session:
            if session.get(KnowledgeBase, kb_id) is not None:
                raise KnowledgeBaseError("Exista deja o baza de cunostinte cu acest id.")
            kb = KnowledgeBase(id=kb_id, name=name, description=description)
            session.add(kb)
            session.flush()
            return {"id": kb.id, "name": kb.name, "description": kb.description, "created_at": kb.created_at,
                    "document_count": 0, "qa_record_count": 0}

    def delete(self, kb_id: str) -> None:
        if kb_id == "default":
            raise KnowledgeBaseError("Baza de cunostinte implicita nu poate fi stearsa.")
        with self.db.session() as session:
            kb = session.get(KnowledgeBase, kb_id)
            if kb is None:
                raise KnowledgeBaseError("Baza de cunostinte nu exista.")
            for doc in session.execute(select(Document).where(Document.kb_id == kb_id)).scalars():
                session.delete(doc)
            for ds in session.execute(select(QADataset).where(QADataset.kb_id == kb_id)).scalars():
                session.delete(ds)
            session.delete(kb)
        self.store.drop_knowledge_base(kb_id)
