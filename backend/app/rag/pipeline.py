from dataclasses import dataclass


# Final response returned by the RAG pipeline
@dataclass
class RAGResponse:
    answer: str
    sources: list
    retrieved_count: int
    reranked_count: int


class RAGPipeline:

    def __init__(
        self,
        query_processor,
        retriever,
        reranker,
        context_builder,
        llm
    ):
        # Inject each RAG component into the pipeline
        self.query_processor = query_processor
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm

    def answer_query(
        self,
        query: str,
        document_ids: list[str] | None = None
    ) -> RAGResponse:

        # 1. Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # 2. Process/clean the query
        processed_query = self.query_processor.process(query)

        # 3. Retrieve relevant candidate chunks
        candidates = self.retriever.retrieve(
            query=processed_query,
            document_ids=document_ids
        )

        # Return early if no relevant chunks were found
        if not candidates:
            return RAGResponse(
                answer="I could not find relevant information in the provided documents.",
                sources=[],
                retrieved_count=0,
                reranked_count=0
            )

        # 4. Rerank candidates by relevance
        reranked = self.reranker.rank(
            query=processed_query,
            documents=candidates
        )

        # 5. Build context for the LLM
        context = self.context_builder.build(reranked)

        # 6. Generate answer using the retrieved context
        answer = self.llm.generate(
            query=query,
            context=context
        )

        # 7. Extract citation/source metadata
        sources = self._extract_sources(reranked)

        # Return answer along with retrieval statistics and sources
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_count=len(candidates),
            reranked_count=len(reranked)
        )

    @staticmethod
    def _extract_sources(chunks):

        sources = []

        for chunk in chunks:
            # Get metadata stored with each chunk
            metadata = chunk.get("metadata", {})

            sources.append({
<<<<<<< HEAD
                "document_id": metadata.get("document_id"),
                "page": metadata.get("page"),
                "section": metadata.get("section")
=======
                "document_id": metadata.get("doc_id"),
                "page": metadata.get("page_number"),
                "section": metadata.get("section_title")
>>>>>>> 4669588 (add retrieval cache and citations)
            })

        return sources