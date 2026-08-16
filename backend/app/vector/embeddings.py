from sentence_transformers import SentenceTransformer  # HuggingFace library for embeddings

# BGE-base-en-v1.5 produces 768-dimensional embeddings — loaded once at import time
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed(text: str) -> list[float]:
    """Convert a single text string into a 768-dimensional embedding vector."""
    embedding = model.encode(text)  # run the model on the text
    return embedding.tolist()  # convert numpy array to plain Python list for JSON/Qdrant


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convert a list of texts into embeddings in one batch — much faster than one at a time."""
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)  # batch encode
    return embeddings.tolist()
