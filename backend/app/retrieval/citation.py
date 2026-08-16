def create_citation(chunk):
    """Build a citation dict from a single chunk's metadata payload."""
    metadata = chunk.get("payload", {})  # Qdrant payload or BM25 normalized payload
    return {
        "document": metadata.get("filename"),  # original PDF filename
        "page": metadata.get("page_number"),  # source page for this chunk
        "section": metadata.get("section_title"),  # heading above this chunk
        "chunk_id": metadata.get("chunk_id"),  # stable chunk identifier
        "score": chunk.get("rerank_score") or chunk.get("score"),  # prefer rerank score if available
    }


def create_citations(chunks):
    """Convert a list of chunks into a list of citation dicts."""
    citations = []
    for chunk in chunks:
        citations.append(create_citation(chunk))
    return citations
