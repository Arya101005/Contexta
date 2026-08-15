from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import ChatSession, Document, Message


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
    Upload a document and store its metadata in PostgreSQL.
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

    return {
        "message": "Document uploaded successfully",
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status
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
    Delete a document from PostgreSQL.
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
    Create a chat message.

    IMPORTANT:
    This backend stores the question/answer.
    The actual RAG/LLM answer generation belongs
    to the other team members.
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

    return {
        "session_id": session_id,
        "message_id": message.id,
        "question": message.question,
        "answer": message.answer,
        "status": "question_stored"
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