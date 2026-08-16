import logging  # structured logging for debugging and timing
from pathlib import Path  # filesystem path handling

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile  # web framework pieces
from sqlalchemy.orm import Session  # database session type

from backend.app.database import get_db  # dependency that yields a DB session
from backend.app.models.models import ChatSession, Document, Message  # ORM models
from backend.app.rag import build_pipeline, ingest_pdf  # RAG pipeline + PDF ingestion
from backend.app.vector.qdrant import delete_by_doc_id  # remove vectors from Qdrant
from backend.app.retrieval.retriever import rebuild_bm25  # rebuild sparse index after changes

logger = logging.getLogger(__name__)  # logger named after this module

router = APIRouter()  # all routes in this file are grouped under this router


# ==========================
# DOCUMENT UPLOAD
# ==========================

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),  # uploaded PDF file from the frontend
    db: Session = Depends(get_db)  # inject a DB session for this request
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    upload_dir = Path("uploads")  # save uploaded files to ./uploads/
    upload_dir.mkdir(exist_ok=True)  # create folder if it doesn't exist

    file_path = upload_dir / file.filename  # full path where we'll save the PDF

    contents = await file.read()  # read the entire uploaded file into memory

    with open(file_path, "wb") as buffer:  # write bytes to disk
        buffer.write(contents)

    # create a Document record in PostgreSQL with status "uploaded"
    document = Document(filename=file.filename, file_path=str(file_path), status="uploaded")
    db.add(document)
    db.commit()
    db.refresh(document)

    document.status = "processing"  # mark as in-progress while we parse/index
    db.commit()

    try:
        # run the full ingestion pipeline: parse -> chunk -> PostgreSQL -> embed -> Qdrant -> BM25
        chunks, doc_id = ingest_pdf(str(file_path), db_session=db, document_id=document.id)
        document.doc_id = doc_id  # store the SHA-256 doc_id for Qdrant filtering
        document.page_count = len(chunks)  # approximate page count from chunks
        document.status = "ready"  # ingestion succeeded
        db.commit()
        db.refresh(document)

        return {
            "message": "Document uploaded successfully",
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "chunks_count": len(chunks)
        }
    except Exception as e:
        document.status = "failed"  # mark as failed so frontend can show error
        document.error = str(e)  # preserve error message in DB for debugging
        db.commit()
        logger.exception("Failed to process document %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# ==========================
# LIST DOCUMENTS
# ==========================

@router.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.uploaded_at.desc()).all()  # newest first

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "doc_id": document.doc_id,
            "file_path": document.file_path,
            "uploaded_at": document.uploaded_at,
            "status": document.status,
            "page_count": document.page_count,
            "error": document.error,
        }
        for document in documents
    ]


# ==========================
# DELETE DOCUMENT
# ==========================

@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_id = document.doc_id

    # delete all Qdrant vectors for this document first
    if doc_id:
        delete_by_doc_id(doc_id)

    # delete the physical PDF file from disk
    if document.file_path:
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()

    # delete the document record from PostgreSQL (cascades to chunks)
    db.delete(document)
    db.commit()

    rebuild_bm25(db)  # rebuild sparse index after removal

    return {"message": "Document deleted successfully", "document_id": document_id}


# ==========================
# CHAT / RAG QUERY
# ==========================

@router.post("/chat")
def create_chat(
    question: str,  # user's question from the frontend
    session_id: int | None = None,  # optional existing chat session
    document_id: int | None = None,  # optional document to scope the search
    db: Session = Depends(get_db)
):
    # create a new session if the frontend didn't provide one
    if session_id is None:
        session = ChatSession()
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

    # resolve the selected document (if any) to get its doc_id for scoped retrieval
    document = None
    doc_ids = None
    if document_id is not None:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.doc_id:
            doc_ids = [document.doc_id]  # pass Qdrant-scoped doc_id to the pipeline

    # save the user's question to the messages table
    message = Message(session_id=session_id, document_id=document_id, question=question, answer=None)
    db.add(message)
    db.commit()
    db.refresh(message)

    try:
        # build the RAG pipeline (lazy, so models load only when first query arrives)
        pipeline = build_pipeline()
        # run the full pipeline: retrieve -> fuse -> rerank -> expand -> LLM
        response = pipeline.answer_query(question, document_ids=doc_ids)
        message.answer = response.answer
        db.commit()
        db.refresh(message)
    except Exception as e:
        message.answer = f"Error generating answer: {str(e)}"
        db.commit()
        db.refresh(message)
        logger.exception("RAG pipeline failed for session %s", session_id)

    return {
        "session_id": session_id,
        "message_id": message.id,
        "question": message.question,
        "answer": message.answer,
        "sources": response.sources if "response" in locals() else [],
        "status": "answered"
    }


# ==========================
# GET CHAT HISTORY
# ==========================

@router.get("/chat/{session_id}")
def get_chat_history(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())  # oldest message first
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
