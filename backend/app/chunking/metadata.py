"""
backend/app/chunking/metadata.py
Schema and contextual metadata enrichment for vector indexing.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid


class ChunkPayload(BaseModel):
    """Schema for an enriched text chunk ready for vector embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    filename: str
    page_number: int
    section_title: str = "General"
    content_type: str = "text"  # 'text' or 'table'
    token_count: int
    char_count: int
    text: str
    context_prefix: str = ""

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Converts chunk model to a standard Qdrant payload dictionary."""
        return self.model_dump()


class MetadataEnricher:
    """Attaches contextual prefixes and metadata tags."""

    @staticmethod
    def create_payload(
        doc_id: str,
        filename: str,
        page_number: int,
        text: str,
        section_title: str,
        content_type: str,
        token_count: int
    ) -> ChunkPayload:
        context_prefix = f"Document: {filename} | Section: {section_title} | Page: {page_number}\n"
        return ChunkPayload(
            doc_id=doc_id,
            filename=filename,
            page_number=page_number,
            section_title=section_title,
            content_type=content_type,
            token_count=token_count,
            char_count=len(text),
            text=text,
            context_prefix=context_prefix
        )