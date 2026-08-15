from sentence_transformers import SentenceTransformer

# BGE-base-en-v1.5 produces 768-dimensional embeddings.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed(text: str) -> list[float]:
    """
    Convert text into a 768-dimensional embedding.
    """
    embedding = model.encode(text)

    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a batch of texts into embeddings.
    """
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()