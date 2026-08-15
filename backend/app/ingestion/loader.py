"""
backend/app/ingestion/loader.py
Loads PDF files from disk or memory streams and calculates unique SHA-256 IDs.
"""

from pathlib import Path
from typing import Dict, Any, Union
import hashlib
import pymupdf


class DocumentLoader:
    """Loads and validates PDF documents for the ingestion pipeline."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Generates a SHA-256 hash to uniquely identify documents."""
        return hashlib.sha256(file_bytes).hexdigest()

    @classmethod
    def load_from_path(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads a PDF from local storage."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {path.suffix}. Expected PDF.")

        file_bytes = path.read_bytes()
        doc_id = cls.compute_file_hash(file_bytes)

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        metadata = {
            "doc_id": doc_id,
            "filename": path.name,
            "page_count": len(doc),
            "raw_metadata": doc.metadata,
            "file_size_kb": round(len(file_bytes) / 1024, 2)
        }
        doc.close()
        return {"doc_id": doc_id, "file_bytes": file_bytes, "metadata": metadata}