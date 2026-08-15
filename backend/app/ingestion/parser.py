"""
backend/ingestion/parser.py
Coordinates page parsing and OCR fallback.
"""

from typing import List, Dict, Any
import pymupdf
from backend.app.ingestion.layout import LayoutAnalyzer


class DocumentParser:
    """Parses complete PDF streams into layout elements per page."""

    def __init__(self, layout_analyzer: LayoutAnalyzer = None):
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()

    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[Dict[str, Any]]:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        parsed_pages = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text().strip()

            if len(raw_text) < self.layout_analyzer.min_char_threshold:
                ocr_text = self.layout_analyzer.ocr_page(page)
                elements = [{
                    "type": "body",
                    "text": ocr_text,
                    "bbox": list(page.rect),
                    "font_size": 10,
                    "is_bold": False
                }]
            else:
                try:
                    elements = self.layout_analyzer.extract_page_layout(page)
                except Exception:
                    elements = [{
                        "type": "body",
                        "text": raw_text,
                        "bbox": list(page.rect),
                        "font_size": 10,
                        "is_bold": False
                    }]

            parsed_pages.append({
                "doc_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "elements": elements
            })

        doc.close()
        return parsed_pages