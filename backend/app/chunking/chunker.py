"""
backend/app/chunking/chunker.py
Splits document elements into semantic chunks while maintaining structural context.
"""

import re
from typing import List, Dict, Any
from backend.app.chunking.metadata import ChunkPayload, MetadataEnricher


class SmartChunker:
    """
    Semantic chunker optimized for BGE-base (768-dim, 512 max tokens):
    - Tables are preserved as atomic chunks.
    - Headers update the active section context.
    - Text blocks are chunked on sentence boundaries with overlap.
    """

    def __init__(self, target_chunk_tokens: int = 350, max_chunk_tokens: int = 512, overlap_tokens: int = 50):
        self.target_tokens = target_chunk_tokens
        self.max_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast heuristic token counter (~1.3 tokens per whitespace-delimited word)."""
        return max(1, int(len(text.split()) * 1.3))

    def split_sentences(self, text: str) -> List[str]:
        """Splits text on punctuation boundaries."""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, parsed_pages: List[Dict[str, Any]]) -> List[ChunkPayload]:
        all_chunks: List[ChunkPayload] = []
        current_section = "Introduction"

        for page in parsed_pages:
            doc_id = page["doc_id"]
            filename = page["filename"]
            page_num = page["page_number"]
            elements = page["elements"]

            current_text_buffer = []
            current_buffer_tokens = 0

            for el in elements:
                el_type = el.get("type", "body")
                el_text = el.get("text", "").strip()
                if not el_text:
                    continue

                # 1. Header Handling: Update active section context and flush buffer
                if el_type in ("header_1", "header_2"):
                    if current_text_buffer:
                        combined = " ".join(current_text_buffer)
                        all_chunks.append(
                            MetadataEnricher.create_payload(
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=combined,
                                section_title=current_section,
                                content_type="text",
                                token_count=current_buffer_tokens
                            )
                        )
                        current_text_buffer = []
                        current_buffer_tokens = 0
                    current_section = el_text
                    continue

                # 2. Table Handling: Preserve as an atomic chunk
                if el_type == "table":
                    if current_text_buffer:
                        combined = " ".join(current_text_buffer)
                        all_chunks.append(
                            MetadataEnricher.create_payload(
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=combined,
                                section_title=current_section,
                                content_type="text",
                                token_count=current_buffer_tokens
                            )
                        )
                        current_text_buffer = []
                        current_buffer_tokens = 0

                    table_tokens = self.estimate_tokens(el_text)
                    all_chunks.append(
                        MetadataEnricher.create_payload(
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_num,
                            text=el_text,
                            section_title=current_section,
                            content_type="table",
                            token_count=table_tokens
                        )
                    )
                    continue

                # 3. Standard Body Text Handling with Sentence Splitting & Overlap
                sentences = self.split_sentences(el_text)
                for sentence in sentences:
                    sentence_tokens = self.estimate_tokens(sentence)

                    if current_buffer_tokens + sentence_tokens > self.max_tokens:
                        combined = " ".join(current_text_buffer)
                        all_chunks.append(
                            MetadataEnricher.create_payload(
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                text=combined,
                                section_title=current_section,
                                content_type="text",
                                token_count=current_buffer_tokens
                            )
                        )

                        # Window overlap
                        overlap_buffer = []
                        overlap_count = 0
                        for prev_s in reversed(current_text_buffer):
                            tokens = self.estimate_tokens(prev_s)
                            if overlap_count + tokens <= self.overlap_tokens:
                                overlap_buffer.insert(0, prev_s)
                                overlap_count += tokens
                            else:
                                break

                        current_text_buffer = overlap_buffer + [sentence]
                        current_buffer_tokens = overlap_count + sentence_tokens
                    else:
                        current_text_buffer.append(sentence)
                        current_buffer_tokens += sentence_tokens

            # Flush remaining buffer at the end of the page
            if current_text_buffer:
                combined = " ".join(current_text_buffer)
                all_chunks.append(
                    MetadataEnricher.create_payload(
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_num,
                        text=combined,
                        section_title=current_section,
                        content_type="text",
                        token_count=current_buffer_tokens
                    )
                )

        return all_chunks