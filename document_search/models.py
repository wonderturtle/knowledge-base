"""SQLAlchemy models for Document Search."""

from datetime import datetime, timezone
import json
from . import db


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    original_name = db.Column(db.String, nullable=False)
    storage_name = db.Column(db.String, nullable=False, unique=True)
    content_hash = db.Column(db.String, nullable=False)
    file_type = db.Column(db.String, nullable=False)
    parser_used = db.Column(db.String, nullable=False, default="")
    extracted_sections = db.Column(db.Text, nullable=True)
    extracted_text = db.Column(db.Text, nullable=False, default="")
    extraction_warnings = db.Column(db.Text, nullable=False, default="")
    status = db.Column(db.String, nullable=False, default="pending")
    page_count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    passages = db.relationship(
        "Passage",
        backref="document",
        cascade="all, delete-orphan",
        order_by="Passage.seq",
        lazy=True,
    )

    @property
    def warnings_list(self):
        if not self.extraction_warnings:
            return []
        try:
            val = json.loads(self.extraction_warnings)
            return val if isinstance(val, list) else [str(val)]
        except Exception:
            return [self.extraction_warnings]

    def to_dict(self):
        return {
            "id": self.id,
            "original_name": self.original_name,
            "storage_name": self.storage_name,
            "content_hash": self.content_hash,
            "file_type": self.file_type,
            "parser_used": self.parser_used,
            "extracted_text": self.extracted_text,
            "extraction_warnings": self.warnings_list,
            "status": self.status,
            "page_count": self.page_count,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
        }


class Passage(db.Model):
    __tablename__ = "passages"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    document_id = db.Column(
        db.Integer,
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    page_or_section = db.Column(db.String, nullable=False)
    embedding = db.Column(db.Text, nullable=True)
    model_id = db.Column(db.String, nullable=True)
    embedding_dim = db.Column(db.Integer, nullable=True)

    __table_args__ = (
        db.UniqueConstraint("document_id", "seq", name="uq_doc_seq"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "seq": self.seq,
            "text": self.text,
            "page_or_section": self.page_or_section,
            "embedding": self.embedding,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
        }
