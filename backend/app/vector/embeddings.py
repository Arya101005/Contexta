from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def embed(text):
    return model.encode(text).tolist()