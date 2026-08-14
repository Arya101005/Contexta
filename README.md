# Contexta

Contexta is a structure-aware, retrieval-augmented Q&A system designed to deliver high-precision answers with exact document citations.

## Features

- Document ingestion with layout-aware parsing
- Semantic chunking with rich metadata
- Vector storage and retrieval via Qdrant
- Hybrid retrieval with reranking
- LLM-powered answer generation with guardrails
- End-to-end RAG pipeline
- Evaluation framework for retrieval and generation quality
- React frontend for document upload, chat, and source inspection

## Architecture

```
backend/
  app/
    ingestion/   → load, parse, and layout documents
    chunking/    → split documents into chunks with metadata
    vector/      → embeddings and Qdrant vector store
    retrieval/   → retriever and reranker
    llm/         → LLM client, prompts, and guardrails
    rag/         → end-to-end RAG pipeline
    models/      → Pydantic models
    evaluation/  → retrieval/generation evaluators
frontend/
  src/
    components/  → Upload, Chat, Sources
tests/
  test_ingestion.py, test_chunking.py, test_vector.py,
  test_retrieval.py, test_rag.py
```

## Tech Stack

- Backend: FastAPI, Python
- Vector DB: Qdrant
- LLM: OpenAI / compatible providers
- Frontend: React
- Testing: pytest

## Getting Started

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## API Endpoints

- `POST /api/upload` — upload documents
- `POST /api/chat` — send a query and get a RAG answer
- `GET /api/sources/{query_id}` — retrieve cited sources

## Testing

```bash
pytest tests/ -v
```

## License

MIT
