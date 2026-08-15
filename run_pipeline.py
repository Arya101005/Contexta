"""
run_pipeline.py
Executes the document ingestion and chunking pipeline on the test PDF.
"""

from pathlib import Path
from backend.app.ingestion.loader import DocumentLoader
from backend.app.ingestion.parser import DocumentParser
from backend.app.chunking.chunker import SmartChunker


def run():
    pdf_path = Path("golden_test_report.pdf")
    if not pdf_path.exists():
        pdf_path = Path("docs/golden_test_report.pdf")

    if not pdf_path.exists():
        print("[!] Error: Could not find golden_test_report.pdf")
        return

    print("=" * 70)
    print("STEP 1: DOCUMENT INGESTION & LOADING")
    print("=" * 70)
    loader = DocumentLoader()
    doc_info = loader.load_from_path(pdf_path)
    print(f"[*] Document ID (SHA-256): {doc_info['doc_id']}")
    print(f"[*] Filename:              {doc_info['metadata']['filename']}")
    print(f"[*] Total Pages:           {doc_info['metadata']['page_count']}")
    print(f"[*] File Size:             {doc_info['metadata']['file_size_kb']} KB\n")

    print("=" * 70)
    print("STEP 2: DOCUMENT AI - LAYOUT ANALYSIS & PARSING")
    print("=" * 70)
    parser = DocumentParser()
    parsed_pages = parser.parse(
        file_bytes=doc_info["file_bytes"],
        doc_id=doc_info["doc_id"],
        filename=doc_info["metadata"]["filename"]
    )
    for p in parsed_pages:
        element_types = [e["type"] for e in p["elements"]]
        print(f"[*] Page {p['page_number']}: Found {len(p['elements'])} elements -> {element_types}")

    print("\n" + "=" * 70)
    print("STEP 3: SMART CHUNKING & METADATA ENRICHMENT")
    print("=" * 70)
    chunker = SmartChunker(target_chunk_tokens=350, max_chunk_tokens=512, overlap_tokens=50)
    chunks = chunker.chunk_document(parsed_pages)
    print(f"[*] Total Vector Payloads Generated: {len(chunks)}\n")

    print("=" * 70)
    print("OUTPUT PAYLOADS READY FOR EMBEDDING TEAM (BGE-base 768-dim)")
    print("=" * 70)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n--- [CHUNK #{idx}] | Type: {chunk.content_type.upper()} | Page: {chunk.page_number} ---")
        print(f"Section Context : {chunk.section_title}")
        print(f"Token Count     : {chunk.token_count} tokens")
        print(f"Prefix Injected :\n{chunk.context_prefix}")
        print(f"Text Payload    :\n{chunk.text}")
        print("-" * 50)


if __name__ == "__main__":
    run()