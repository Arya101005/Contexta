from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router
from backend.app.database import Base, engine
from backend.app.models import models

from backend.app.api import router


app = FastAPI(
    title="Contexta API",
    description="Backend API for the Contexta RAG system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Contexta backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
