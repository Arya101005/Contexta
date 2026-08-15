def create_citation(chunk):
    """
    Create a citation from chunk metadata.
    """

    metadata = chunk["metadata"]

    return {
        "document": metadata.get("filename"),
        "page": metadata.get("page_number"),
        "section": metadata.get("section_title"),
        "score": chunk.get("rerank_score")
    }


def create_citations(chunks):

    citations = []

    for chunk in chunks:
        citations.append(
            create_citation(chunk)
        )

    return citations