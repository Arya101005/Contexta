from fastapi import FastAPI

from backend.app.api import router


app = FastAPI(
    title="Contexta API",
    description="Backend API for the Contexta RAG system",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Contexta backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
