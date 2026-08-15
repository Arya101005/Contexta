from sentence_transformers import SentenceTransformer

# BGE-base-en-v1.5 produces 768-dimensional vectors.
# normalize_embeddings=True makes cosine similarity work nicely.
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed(text: str) -> list[float]:
    # Convert text -> 768-dimensional embedding vector.
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()