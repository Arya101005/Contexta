from datetime import datetime  # for created_at timestamps

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Float, Index
from sqlalchemy.orm import relationship  # defines relationships between tables

from backend.app.database import Base  # all models inherit from this


class Document(Base):
    __tablename__ = "documents"  # table name in PostgreSQL

    id = Column(Integer, primary_key=True, index=True)  # auto-incrementing primary key
    filename = Column(String(255), nullable=False)  # original uploaded filename
    doc_id = Column(String(64), unique=True, nullable=True)  # SHA-256 hash, unique across all docs
    file_path = Column(String(500), nullable=True)  # local filesystem path to saved PDF
    uploaded_at = Column(DateTime, default=datetime.utcnow)  # when the upload happened
    status = Column(String(50), default="uploaded")  # uploaded -> processing -> ready/failed
    page_count = Column(Integer, nullable=True)  # number of pages in the PDF
    error = Column(Text, nullable=True)  # stores error message if ingestion fails

    # one document has many messages (chat history)
    messages = relationship("Message", back_populates="document")
    # one document has many chunks (the actual text pieces)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    # cascade="all, delete-orphan" = deleting a document also deletes all its chunks


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # one session has many messages
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)  # link to chat session
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)  # optional link to a document

    question = Column(Text, nullable=False)  # user's question
    answer = Column(Text, nullable=True)  # AI-generated answer
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
    document = relationship("Document", back_populates="messages")


class Chunk(Base):
    __tablename__ = "chunks"  # stores every text chunk from every uploaded document

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(64), unique=True, nullable=False, index=True)  # stable UUID string
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)  # which document
    filename = Column(String(255), nullable=False)  # cached filename for quick access
    page_number = Column(Integer, nullable=True)  # primary source page for citations
    start_page = Column(Integer, nullable=True)  # first page this chunk spans (can cross pages)
    end_page = Column(Integer, nullable=True)  # last page this chunk spans
    section_title = Column(String(255), nullable=True)  # immediate heading above this chunk
    section_path = Column(String(1024), nullable=True, index=True)  # full hierarchy path
    parent_section = Column(String(255), nullable=True)  # one level up in hierarchy
    content_type = Column(String(50), nullable=False, index=True)  # "text", "table", "image_ocr"
    chunk_index = Column(Integer, nullable=False)  # order of this chunk within its document
    previous_chunk_id = Column(String(64), nullable=True)  # for context expansion
    next_chunk_id = Column(String(64), nullable=True)  # for context expansion
    token_count = Column(Integer, nullable=True)  # estimated tokens in this chunk
    char_count = Column(Integer, nullable=True)  # exact character count
    text = Column(Text, nullable=False)  # the actual chunk text content
    context_prefix = Column(Text, nullable=True)  # prepended metadata string for the LLM
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


# composite index for fast queries like "get all text chunks for document X"
Index("ix_chunks_document_content", Chunk.document_id, Chunk.content_type)
