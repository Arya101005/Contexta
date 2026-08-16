"""
backend/app/ingestion/layout.py
Analyzes page layout: extracts tables, headers, body text, and image OCR.
Sorts elements top-to-bottom so chunks respect reading order.
"""

from typing import List, Dict, Any, Optional  # type hints
import io  # in-memory byte stream for image conversion
import pymupdf  # PyMuPDF — table detection, rendering, image extraction
from PIL import Image  # convert rendered page/image to PIL format for Tesseract
import pytesseract  # OCR engine — only used when native text is insufficient


class LayoutAnalyzer:
    """Extracts layout elements: structural headers, body text, tables, and OCR-derived image text."""

    def __init__(self, min_char_threshold: int = 50, tesseract_cmd: Optional[str] = None):
        self.min_char_threshold = min_char_threshold  # pages with fewer chars get OCR fallback
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd  # custom Tesseract path on Windows

    def _is_tiny_or_decorative(self, img_rect, page_rect) -> bool:
        """
        Return True if the image should be skipped entirely.
        Skip only if BOTH:
          - extremely tiny (< 0.3% of page) — icons, bullets, dots
          - in page margins (top/bottom 10%) — likely logos/watermarks/header graphics
        Charts and graphs often pass both checks, so they get OCR'd.
        """
        area = img_rect.width * img_rect.height
        page_area = page_rect.width * page_rect.height
        if page_area <= 0:
            return True
        ratio = area / page_area

        # skip truly tiny images (icons, bullets, decorative dots)
        if ratio < 0.003:
            return True

        # skip images in page margins — logos/watermarks often sit in header/footer
        margin_threshold = page_rect.height * 0.10
        in_top_margin = img_rect.y1 < margin_threshold
        in_bottom_margin = img_rect.y0 > (page_rect.height - margin_threshold)
        if in_top_margin or in_bottom_margin:
            return True

        return False

    def extract_page_layout(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Extract tables, headers, and body text from a single page."""
        layout_elements = []

        # 1. Extract tables and remember their bounding boxes so we don't duplicate their text later
        table_bboxes = []
        try:
            tables = page.find_tables()  # PyMuPDF built-in table detector
            for tab in tables:
                table_bboxes.append(pymupdf.Rect(tab.bbox))  # store table rectangle for dedup
                df = tab.extract()  # extract table as list of lists
                if df:
                    headers = [str(c) if c is not None else "" for c in df[0]]
                    md_table = "| " + " | ".join(headers) + " |\n"  # markdown header row
                    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"  # markdown separator
                    for row in df[1:]:
                        md_table += "| " + " | ".join([str(c) if c is not None else "" for c in row]) + " |\n"

                    layout_elements.append({
                        "type": "table",  # content type tag
                        "text": md_table.strip(),  # table content as markdown string
                        "bbox": tab.bbox,
                        "font_size": 10,
                        "is_bold": False
                    })
        except Exception:
            pass  # if table detection fails, continue with text extraction

        # 2. Extract text blocks and classify lines as header_1, header_2, or body
        blocks = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE)["blocks"]  # dehyphenate joined words
        for block in blocks:
            if block.get("type") == 0:  # type 0 = text block (type 1 = image)
                block_rect = pymupdf.Rect(block.get("bbox"))

                # skip text that overlaps with a table — avoids duplicating table content
                if any(block_rect.intersects(tb) for tb in table_bboxes):
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue  # skip empty lines

                    spans = line.get("spans", [])
                    max_font_size = max(span.get("size", 10) for span in spans) if spans else 10
                    is_bold = any(bool(span.get("flags", 0) & 2 or "bold" in span.get("font", "").lower()) for span in spans)

                    content_type = "body"
                    if max_font_size >= 16 or (max_font_size >= 13 and is_bold):
                        content_type = "header_1"  # large or bold = top-level heading
                    elif max_font_size >= 12 and is_bold:
                        content_type = "header_2"  # slightly smaller bold = sub-heading

                    layout_elements.append({
                        "type": content_type,
                        "text": line_text,
                        "bbox": line.get("bbox"),
                        "font_size": max_font_size,
                        "is_bold": is_bold
                    })

        # 3. Sort all elements top-to-bottom so chunk order matches visual reading order
        def get_y_coordinate(element):
            if element.get("bbox"):
                return element["bbox"][1]  # bbox[1] = top y-coordinate
            return float('inf')  # elements without bbox go to the end

        layout_elements.sort(key=get_y_coordinate)
        return layout_elements

    def extract_images(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Extract images from a page and OCR text-heavy ones."""
        image_elements = []
        page_rect = page.rect

        try:
            img_list = page.get_images(full=True)  # get all embedded images on the page
            for img_index, img in enumerate(img_list):
                xref = img[0]  # internal PDF reference number for the image
                pix = pymupdf.Pixmap(page.parent, xref)  # render image to pixmap
                if pix.n > 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)  # convert CMYK to RGB if needed

                # find the bounding box of this image on the page
                img_rect = None
                for img_info in page.get_image_info():
                    if img_info.get("xref") == xref:
                        img_rect = pymupdf.Rect(img_info.get("bbox", [0, 0, 0, 0]))
                        break
                if img_rect is None:
                    img_rect = pymupdf.Rect(0, 0, pix.width, pix.height)

                # skip tiny/decorative images to save OCR time
                if self._is_tiny_or_decorative(img_rect, page_rect):
                    pix = None
                    continue

                try:
                    pil_img = Image.open(io.BytesIO(pix.tobytes("png")))  # convert pixmap to PIL Image
                    ocr_text = pytesseract.image_to_string(pil_img).strip()  # run OCR
                except Exception:
                    ocr_text = ""
                finally:
                    pix = None  # free memory

                if ocr_text:  # only keep images that produced readable text
                    image_elements.append({
                        "type": "image_ocr",
                        "text": ocr_text,
                        "bbox": list(img_rect),
                        "font_size": 10,
                        "is_bold": False
                    })

        except Exception:
            pass  # if image extraction fails entirely, continue without images

        return image_elements

    def ocr_page(self, page: pymupdf.Page, dpi: int = 300) -> str:
        """Fallback OCR for scanned pages with no native text."""
        try:
            pix = page.get_pixmap(dpi=dpi)  # render entire page as image at 300 DPI
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img).strip()  # run OCR and return text
        except Exception:
            return ""  # return empty string if OCR fails
