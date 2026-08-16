"""
backend/app/ingestion/loader.py
Reads a PDF from disk and produces a unique SHA-256 doc_id + raw bytes.
"""

from pathlib import Path  # cross-platform filesystem paths
from typing import Dict, Any, Union  # type hints
import hashlib  # for SHA-256 hashing
import pymupdf  # PyMuPDF — native PDF text extraction


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf"}  # only PDFs are accepted right now

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """SHA-256 hash of the entire file — used as a stable document ID."""
        return hashlib.sha256(file_bytes).hexdigest()

    @classmethod
    def load_from_path(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a PDF from disk, return bytes + metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {path.suffix}. Expected PDF.")

        file_bytes = path.read_bytes()  # read entire file into memory
        doc_id = cls.compute_file_hash(file_bytes)  # compute stable document ID

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")  # open PDF to read metadata
        metadata = {
            "doc_id": doc_id,
            "filename": path.name,
            "page_count": len(doc),  # total number of pages
            "raw_metadata": doc.metadata,  # PDF metadata dict (author, title, etc.)
            "file_size_kb": round(len(file_bytes) / 1024, 2)  # file size in KB
        }
        doc.close()
        return {"doc_id": doc_id, "file_bytes": file_bytes, "metadata": metadata}
