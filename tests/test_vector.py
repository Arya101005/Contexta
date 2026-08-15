from backend.app.vector.embeddings import embed

def test_embedding():
    vector = embed("Revenue increased by 15 percent.")
<<<<<<< HEAD
    assert len(vector) == 1024
=======
    assert len(vector) == 768
>>>>>>> 4669588 (add retrieval cache and citations)
