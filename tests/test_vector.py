from backend.app.vector.embeddings import embed

def test_embedding():
    vector = embed("Revenue increased by 15 percent.")
    assert len(vector) == 1024