"""
backend/app/ingestion/parser.py
Coordinates page-by-page parsing: native text extraction, OCR fallback, and image OCR.
"""

from typing import List, Dict, Any  # type hints
import pymupdf  # PDF parsing library
from backend.app.ingestion.layout import LayoutAnalyzer  # layout + OCR logic


class DocumentParser:
    """Parses complete PDF streams into layout elements per page."""

    def __init__(self, layout_analyzer: LayoutAnalyzer = None):
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()  # create default if none provided

    def parse(self, file_bytes: bytes, doc_id: str, filename: str) -> List[Dict[str, Any]]:
        """Parse every page in the PDF and return a list of page dicts with elements."""
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")  # open PDF from bytes
        parsed_pages = []

        for page_idx in range(len(doc)):  # iterate over every page
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-based page numbering for human readability
            raw_text = page.get_text().strip()  # extract native text layer from PDF

            elements: List[Dict[str, Any]] = []  # all layout elements for this page

            if len(raw_text) < self.layout_analyzer.min_char_threshold:
                # page has almost no native text — likely a scanned page, use OCR instead
                ocr_text = self.layout_analyzer.ocr_page(page)
                if ocr_text:
                    elements.append({
                        "type": "body",
                        "text": ocr_text,
                        "bbox": list(page.rect),
                        "font_size": 10,
                        "is_bold": False
                    })
            else:
                # page has enough text — extract structured layout (headers, tables, body)
                try:
                    elements = self.layout_analyzer.extract_page_layout(page)
                except Exception:
                    # if layout extraction fails, fall back to raw text as one body block
                    elements = [{
                        "type": "body",
                        "text": raw_text,
                        "bbox": list(page.rect),
                        "font_size": 10,
                        "is_bold": False
                    }]

            # also extract OCR text from any embedded images on this page
            try:
                image_elements = self.layout_analyzer.extract_images(page)
                elements.extend(image_elements)  # add image OCR results to the element list
            except Exception:
                pass  # image extraction is optional, don't fail the whole page

            parsed_pages.append({
                "doc_id": doc_id,  # SHA-256 document identifier
                "filename": filename,  # original filename
                "page_number": page_num,  # which page this is
                "elements": elements  # list of text/table/image elements in reading order
            })

        doc.close()  # free the PDF object
        return parsed_pages
