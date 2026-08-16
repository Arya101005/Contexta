"""
backend/app/chunking/chunker.py
Fast structure-aware hierarchical sentence-aware token-constrained chunking.
- Headers build a section hierarchy path.
- Tables are preserved as atomic chunks.
- Body text is split on sentence boundaries with overlap.
- Chunks can span page boundaries (no forced page flush).
"""

import re  # regex for sentence splitting
from typing import List, Dict, Any, Optional  # type hints
from backend.app.chunking.metadata import ChunkPayload, MetadataEnricher  # chunk schema + factory


class SmartChunker:
    """
    Splits parsed PDF elements into chunks optimized for BGE-base embeddings.
    Target: ~480 tokens, hard max: 512 tokens, overlap: 50 tokens.
    """

    def __init__(
        self,
        target_chunk_tokens: int = 480,  # ideal chunk size (configurable)
        max_chunk_tokens: int = 512,  # hard ceiling — never exceed this
        overlap_tokens: int = 50,  # tokens carried over between adjacent chunks
    ):
        self.target_tokens = target_chunk_tokens
        self.max_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast heuristic: ~1.3 tokens per whitespace-delimited word."""
        return max(1, int(len(text.split()) * 1.3))

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation boundaries."""
        sentences = re.split(r'(?<=[.?!])\s+', text)  # split after . ? or ! followed by whitespace
        return [s.strip() for s in sentences if s.strip()]  # remove empty strings

    def _build_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign section_path and parent_section to each element based on detected headers."""
        hierarchy: List[str] = []  # stack-like list: [section, subsection, subsubsection]
        enriched = []  # output list with hierarchy metadata attached

        for el in elements:
            el_type = el.get("type", "body")
            el_text = el.get("text", "").strip()

            if el_type == "header_1":
                hierarchy = hierarchy[:1]  # truncate to just the top section
                if len(hierarchy) < 1:
                    hierarchy.append(el_text)
                else:
                    hierarchy[0] = el_text  # replace the section name
            elif el_type == "header_2":
                hierarchy = hierarchy[:2]  # truncate to section + subsection
                if len(hierarchy) < 2:
                    hierarchy.append(el_text)
                else:
                    hierarchy[1] = el_text  # replace the subsection name

            section_path = " > ".join(hierarchy) if hierarchy else "General"  # e.g. "Security > Auth"
            parent_section = hierarchy[-2] if len(hierarchy) >= 2 else None  # one level up
            section_title = hierarchy[-1] if hierarchy else "General"  # immediate heading

            enriched.append({
                "type": el_type,
                "text": el_text,
                "bbox": el.get("bbox"),
                "font_size": el.get("font_size", 10),
                "is_bold": el.get("is_bold", False),
                "section_title": section_title,
                "section_path": section_path,
                "parent_section": parent_section,
            })

        return enriched

    def chunk_document(self, parsed_pages: List[Dict[str, Any]]) -> List[ChunkPayload]:
        """Convert parsed pages into a flat list of ChunkPayload objects."""
        all_chunks: List[ChunkPayload] = []
        chunk_index = 0
        previous_chunk_id: Optional[str] = None  # track the last chunk for linking

        for page in parsed_pages:
            doc_id = page["doc_id"]
            filename = page["filename"]
            page_num = page["page_number"]
            elements = page["elements"]

            enriched_elements = self._build_hierarchy(elements)  # attach section metadata

            current_text_buffer: List[str] = []  # sentences accumulated into the current chunk
            current_buffer_tokens = 0  # running token count for the buffer
            current_section_title = "General"
            current_section_path = "General"
            current_parent_section: Optional[str] = None
            chunk_start_page = page_num  # track page span for multi-page chunks
            chunk_end_page = page_num

            for el in enriched_elements:
                el_type = el.get("type", "body")
                el_text = el.get("text", "").strip()
                if not el_text:
                    continue  # skip empty elements

                # update current section context from this element
                current_section_title = el.get("section_title", current_section_title)
                current_section_path = el.get("section_path", current_section_path)
                current_parent_section = el.get("parent_section", current_parent_section)

                # HEADERS: flush current buffer as a chunk, then update section context
                if el_type in ("header_1", "header_2"):
                    if current_text_buffer:
                        combined = " ".join(current_text_buffer)
                        context_prefix = (
                            f"Document: {filename} | Section: {current_section_title} | "
                            f"Path: {current_section_path} | Page: {chunk_start_page}"
                            + (f"–{chunk_end_page}" if chunk_end_page != chunk_start_page else "")
                            + "\n"
                        )
                        payload = MetadataEnricher.create_payload(
                            doc_id=doc_id, filename=filename, page_number=chunk_start_page,
                            text=combined, section_title=current_section_title,
                            section_path=current_section_path, parent_section=current_parent_section,
                            content_type="text", token_count=current_buffer_tokens,
                            context_prefix=context_prefix,
                        )
                        payload.start_page = chunk_start_page
                        payload.end_page = chunk_end_page
                        payload.chunk_index = chunk_index
                        payload.previous_chunk_id = previous_chunk_id
                        all_chunks.append(payload)
                        previous_chunk_id = payload.chunk_id
                        chunk_index += 1
                        current_text_buffer = []
                        current_buffer_tokens = 0
                        chunk_start_page = page_num
                        chunk_end_page = page_num
                    continue  # headers themselves are not stored as chunks

                # TABLES: flush buffer, then store table as one atomic chunk
                if el_type == "table":
                    if current_text_buffer:
                        combined = " ".join(current_text_buffer)
                        context_prefix = (
                            f"Document: {filename} | Section: {current_section_title} | "
                            f"Path: {current_section_path} | Page: {chunk_start_page}"
                            + (f"–{chunk_end_page}" if chunk_end_page != chunk_start_page else "")
                            + "\n"
                        )
                        payload = MetadataEnricher.create_payload(
                            doc_id=doc_id, filename=filename, page_number=chunk_start_page,
                            text=combined, section_title=current_section_title,
                            section_path=current_section_path, parent_section=current_parent_section,
                            content_type="text", token_count=current_buffer_tokens,
                            context_prefix=context_prefix,
                        )
                        payload.start_page = chunk_start_page
                        payload.end_page = chunk_end_page
                        payload.chunk_index = chunk_index
                        payload.previous_chunk_id = previous_chunk_id
                        all_chunks.append(payload)
                        previous_chunk_id = payload.chunk_id
                        chunk_index += 1
                        current_text_buffer = []
                        current_buffer_tokens = 0
                        chunk_start_page = page_num
                        chunk_end_page = page_num

                    table_tokens = self.estimate_tokens(el_text)
                    context_prefix = (
                        f"Document: {filename} | Section: {current_section_title} | "
                        f"Path: {current_section_path} | Page: {page_num}\n"
                    )
                    payload = MetadataEnricher.create_payload(
                        doc_id=doc_id, filename=filename, page_number=page_num,
                        text=el_text, section_title=current_section_title,
                        section_path=current_section_path, parent_section=current_parent_section,
                        content_type="table", token_count=table_tokens,
                        context_prefix=context_prefix,
                    )
                    payload.start_page = page_num
                    payload.end_page = page_num
                    payload.chunk_index = chunk_index
                    payload.previous_chunk_id = previous_chunk_id
                    all_chunks.append(payload)
                    previous_chunk_id = payload.chunk_id
                    chunk_index += 1
                    continue

                # BODY TEXT: split into sentences and accumulate until max tokens reached
                sentences = self.split_sentences(el_text)
                for sentence in sentences:
                    sentence_tokens = self.estimate_tokens(sentence)

                    if current_buffer_tokens + sentence_tokens > self.max_tokens:
                        # buffer is full — flush it as a chunk
                        combined = " ".join(current_text_buffer)
                        context_prefix = (
                            f"Document: {filename} | Section: {current_section_title} | "
                            f"Path: {current_section_path} | Page: {chunk_start_page}"
                            + (f"–{chunk_end_page}" if chunk_end_page != chunk_start_page else "")
                            + "\n"
                        )
                        payload = MetadataEnricher.create_payload(
                            doc_id=doc_id, filename=filename, page_number=chunk_start_page,
                            text=combined, section_title=current_section_title,
                            section_path=current_section_path, parent_section=current_parent_section,
                            content_type="text", token_count=current_buffer_tokens,
                            context_prefix=context_prefix,
                        )
                        payload.start_page = chunk_start_page
                        payload.end_page = chunk_end_page
                        payload.chunk_index = chunk_index
                        payload.previous_chunk_id = previous_chunk_id
                        all_chunks.append(payload)
                        previous_chunk_id = payload.chunk_id
                        chunk_index += 1

                        # overlap: carry over the last N tokens into the next chunk
                        overlap_buffer: List[str] = []
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
                        chunk_start_page = page_num  # new chunk starts here
                        chunk_end_page = page_num
                    else:
                        # sentence fits — add to buffer
                        current_text_buffer.append(sentence)
                        current_buffer_tokens += sentence_tokens
                        chunk_end_page = page_num  # extend chunk's end page

            # NO forced flush at page boundary — chunks can continue across pages

        # flush any remaining buffered text at the very end of the document
        if current_text_buffer:
            combined = " ".join(current_text_buffer)
            context_prefix = (
                f"Document: {filename} | Section: {current_section_title} | "
                f"Path: {current_section_path} | Page: {chunk_start_page}"
                + (f"–{chunk_end_page}" if chunk_end_page != chunk_start_page else "")
                + "\n"
            )
            payload = MetadataEnricher.create_payload(
                doc_id=doc_id, filename=filename, page_number=chunk_start_page,
                text=combined, section_title=current_section_title,
                section_path=current_section_path, parent_section=current_parent_section,
                content_type="text", token_count=current_buffer_tokens,
                context_prefix=context_prefix,
            )
            payload.start_page = chunk_start_page
            payload.end_page = chunk_end_page
            payload.chunk_index = chunk_index
            payload.previous_chunk_id = previous_chunk_id
            all_chunks.append(payload)

        # set next_chunk_id for every chunk (creates a doubly-linked list for context expansion)
        for i, chunk in enumerate(all_chunks):
            if i + 1 < len(all_chunks):
                chunk.next_chunk_id = all_chunks[i + 1].chunk_id

        return all_chunks
