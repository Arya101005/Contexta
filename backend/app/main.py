from fastapi import FastAPI  # the web framework
from fastapi.middleware.cors import CORSMiddleware  # allows frontend to call this API

from backend.app.api import router  # all our route handlers (upload, chat, etc.)
from backend.app.database import Base, engine  # ORM base + DB connection
from backend.app.models import models  # import so SQLAlchemy knows about our tables


app = FastAPI(
    title="Contexta API",  # shown in /docs
    description="Backend API for the Contexta RAG system",
    version="1.0.0"
)

# CORS middleware — allows the React frontend (different port) to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any origin (fine for local dev, restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# auto-create all tables in PostgreSQL if they don't already exist
Base.metadata.create_all(bind=engine)

# include all API routes defined in api.py
app.include_router(router)


@app.get("/")  # health check root endpoint
def root():
    return {"message": "Contexta backend is running"}


@app.get("/health")  # simple liveness probe
def health_check():
    return {"status": "healthy"}


@app.on_event("startup")  # runs once when the server starts
def startup_event():
    # rebuild the in-memory BM25 index from any existing chunks in PostgreSQL
    from backend.app.retrieval.retriever import rebuild_bm25
    from backend.app.database import SessionLocal
    db = SessionLocal()
    try:
        rebuild_bm25(db)
    finally:
        db.close()
