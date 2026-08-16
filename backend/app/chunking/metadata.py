"""
backend/app/chunking/metadata.py
Defines the chunk data schema and enriches chunks with contextual metadata.
"""

from pydantic import BaseModel, Field  # data validation + serialization
from typing import Dict, Any, Optional  # type hints
import uuid  # generate unique chunk IDs


class ChunkPayload(BaseModel):
    """Schema for one enriched text chunk, ready for embedding and storage."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # unique stable ID for this chunk
    doc_id: str  # SHA-256 document ID this chunk belongs to
    filename: str  # original PDF filename
    page_number: Optional[int] = None  # primary source page (for citations)
    start_page: Optional[int] = None  # first page this chunk spans (can cross pages)
    end_page: Optional[int] = None  # last page this chunk spans
    section_title: str = "General"  # immediate heading above this chunk
    section_path: str = ""  # full hierarchy like "Security > Authentication > MFA"
    parent_section: Optional[str] = None  # one level up in the hierarchy
    content_type: str = "text"  # "text", "table", or "image_ocr"
    chunk_index: int = 0  # sequential order of this chunk within its document
    previous_chunk_id: Optional[str] = None  # link to previous chunk for context expansion
    next_chunk_id: Optional[str] = None  # link to next chunk for context expansion
    token_count: int  # estimated token count (for monitoring chunk sizes)
    char_count: int  # exact character count
    text: str  # the actual chunk content
    context_prefix: str = ""  # prepended metadata string like "Document: X | Section: Y | Page: Z"

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert this Pydantic model to a plain dict for Qdrant storage."""
        return self.model_dump()


class MetadataEnricher:
    """Factory for building ChunkPayload objects with correct metadata."""

    @staticmethod
    def create_payload(
        doc_id: str,
        filename: str,
        page_number: int,
        text: str,
        section_title: str,
        section_path: str,
        parent_section: Optional[str],
        content_type: str,
        token_count: int,
        context_prefix: str = "",
    ) -> ChunkPayload:
        """Build a ChunkPayload with all fields populated."""
        return ChunkPayload(
            doc_id=doc_id,
            filename=filename,
            page_number=page_number,
            start_page=page_number,  # will be updated later if chunk spans multiple pages
            end_page=page_number,  # will be updated later if chunk spans multiple pages
            section_title=section_title,
            section_path=section_path,
            parent_section=parent_section,
            content_type=content_type,
            token_count=token_count,
            char_count=len(text),
            text=text,
            context_prefix=context_prefix,
        )
