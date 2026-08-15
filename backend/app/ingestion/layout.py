"""
backend/app/ingestion/layout.py
Analyzes document layout, formats tables, and sorts blocks top-to-bottom.
"""

from typing import List, Dict, Any, Optional
import io
import pymupdf
from PIL import Image
import pytesseract


class LayoutAnalyzer:
    """Extracts layout elements: structural headers, body text, and tables."""

    def __init__(self, min_char_threshold: int = 50, tesseract_cmd: Optional[str] = None):
        self.min_char_threshold = min_char_threshold
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_page_layout(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        layout_elements = []

        # 1. Extract Tables & Map Bounding Boxes
        table_bboxes = []
        try:
            tables = page.find_tables()
            for tab in tables:
                table_bboxes.append(pymupdf.Rect(tab.bbox))
                df = tab.extract()
                if df:
                    headers = [str(c) if c is not None else "" for c in df[0]]
                    md_table = "| " + " | ".join(headers) + " |\n"
                    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in df[1:]:
                        md_table += "| " + " | ".join([str(c) if c is not None else "" for c in row]) + " |\n"

                    layout_elements.append({
                        "type": "table",
                        "text": md_table.strip(),
                        "bbox": tab.bbox,
                        "font_size": 10,
                        "is_bold": False
                    })
        except Exception:
            pass

        # 2. Extract Text Blocks & Headers (Filtering out raw table text)
        blocks = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE)["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # Text block
                block_rect = pymupdf.Rect(block.get("bbox"))

                # Prevent table text duplication
                if any(block_rect.intersects(tb) for tb in table_bboxes):
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue

                    spans = line.get("spans", [])
                    max_font_size = max(span.get("size", 10) for span in spans) if spans else 10
                    is_bold = any(bool(span.get("flags", 0) & 2 or "bold" in span.get("font", "").lower()) for span in spans)

                    content_type = "body"
                    if max_font_size >= 16 or (max_font_size >= 13 and is_bold):
                        content_type = "header_1"
                    elif max_font_size >= 12 and is_bold:
                        content_type = "header_2"

                    layout_elements.append({
                        "type": content_type,
                        "text": line_text,
                        "bbox": line.get("bbox"),
                        "font_size": max_font_size,
                        "is_bold": is_bold
                    })

        # 3. Sort Elements Top-to-Bottom by Y-Coordinate
        def get_y_coordinate(element):
            if element.get("bbox"):
                return element["bbox"][1]
            return float('inf')

        layout_elements.sort(key=get_y_coordinate)
        return layout_elements

    def ocr_page(self, page: pymupdf.Page, dpi: int = 300) -> str:
        """Runs Tesseract OCR on scanned/unselectable pages."""
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""