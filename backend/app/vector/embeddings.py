from sentence_transformers import SentenceTransformer

# BGE-base-en-v1.5 produces 768-dimensional embeddings.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed(text: str) -> list[float]:
    """
    Convert text into a 768-dimensional embedding.
    """
    embedding = model.encode(text)

    return embedding.tolist()