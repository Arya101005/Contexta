from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import ChatSession, Document, Message
from backend.app.rag import build_pipeline, ingest_pdf
from backend.app.vector.qdrant import delete_by_doc_id


router = APIRouter()


# -------------------------
# DOCUMENT APIs
# -------------------------

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document, store its metadata in PostgreSQL,
    and ingest it into the RAG pipeline.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    contents = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    document = Document(
        filename=file.filename,
        file_path=str(file_path),
        status="uploaded"
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # Ingest into RAG pipeline
    try:
        chunks, doc_id = ingest_pdf(str(file_path))
        document.doc_id = doc_id
        document.page_count = len(chunks)
        document.status = "processed"
        db.commit()
        db.refresh(document)
    except Exception as e:
        document.status = "processing_error"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

    return {
        "message": "Document uploaded successfully",
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "chunks_count": len(chunks) if "chunks" in locals() else 0
    }


@router.get("/documents")
def get_documents(
    db: Session = Depends(get_db)
):
    """
    Get all uploaded documents.
    """

    documents = (
        db.query(Document)
        .order_by(Document.uploaded_at.desc())
        .all()
    )

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "doc_id": document.doc_id,
            "file_path": document.file_path,
            "uploaded_at": document.uploaded_at,
            "status": document.status,
            "page_count": document.page_count
        }
        for document in documents
    ]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document from PostgreSQL and its chunks from Qdrant.
    """

    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    # Delete chunks from Qdrant first
    if document.doc_id:
        delete_by_doc_id(document.doc_id)

    # Delete the physical file if it exists
    if document.file_path:
        file_path = Path(document.file_path)

        if file_path.exists():
            file_path.unlink()

    db.delete(document)
    db.commit()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }


# -------------------------
# CHAT APIs
# -------------------------

@router.post("/chat")
def create_chat(
    question: str,
    session_id: int | None = None,
    document_id: int | None = None,
    db: Session = Depends(get_db)
):
    """
    Create a chat message and answer it via the RAG pipeline.
    """

    # Create a new session if one was not provided
    if session_id is None:

        session = ChatSession()

        db.add(session)
        db.commit()
        db.refresh(session)

        session_id = session.id

    else:

        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .first()
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found"
            )

    # Validate document if provided
    if document_id is not None:

        document = (
            db.query(Document)
            .filter(Document.id == document_id)
            .first()
        )

        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

    # Store the question
    message = Message(
        session_id=session_id,
        document_id=document_id,
        question=question,
        answer=None
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    # Answer using the RAG pipeline (built lazily to avoid loading models at import)
    try:
        pipeline = build_pipeline()
        doc_ids = [document.doc_id] if document_id is not None and document else None
        response = pipeline.answer_query(question, document_ids=doc_ids)
        message.answer = response.answer
        db.commit()
        db.refresh(message)
    except Exception as e:
        message.answer = f"Error generating answer: {str(e)}"
        db.commit()
        db.refresh(message)

    return {
        "session_id": session_id,
        "message_id": message.id,
        "question": message.question,
        "answer": message.answer,
        "sources": response.sources if "response" in locals() else [],
        "status": "answered"
    }


@router.get("/chat/{session_id}")
def get_chat_history(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all messages belonging to a chat session.
    """

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
        detail="Chat session not found"
        )

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "session_id": session_id,
        "messages": [
            {
                "message_id": message.id,
                "document_id": message.document_id,
                "question": message.question,
                "answer": message.answer,
                "created_at": message.created_at
            }
            for message in messages
        ]
    }
