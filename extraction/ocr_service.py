"""
Lightweight OCR service for scanned PDFs.

Uses PyMuPDF (fitz) to rasterize pages and pytesseract to perform OCR.
Includes a hybrid mode that prefers native PDF text and falls back to OCR for
pages with little or no extractable text.
"""

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import io

import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps
import pytesseract


class OCRService:
    def __init__(self) -> None:
        # Configure Tesseract path if needed (best-effort for Windows)
        try:
            if os.name == 'nt':
                guess = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                if os.path.isfile(guess):
                    pytesseract.pytesseract.tesseract_cmd = guess
        except Exception:
            pass

    def extract_text_with_ocr(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """Perform OCR on specified pages (1-indexed). If pages is None, OCR all pages.

        Returns dict: { success, pages: [ {page, text, confidence, word_count} ], total_pages, method }
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to open PDF: {e}", "method": "ocr"}

        try:
            total_pages = len(doc)
            if not isinstance(pages, list) or not pages:
                page_indices = list(range(total_pages))
            else:
                page_indices = [max(0, min(total_pages - 1, p - 1)) for p in pages]

            results: List[Dict[str, Any]] = []
            for idx in page_indices:
                try:
                    page = doc[idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    pil_img = self._preprocess_image(pil_img)
                    text = pytesseract.image_to_string(pil_img, config='--psm 6')
                    try:
                        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                        confs = [int(c) for c in data.get('conf', []) if self._is_int(c) and int(c) >= 0]
                        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
                    except Exception:
                        avg_conf = 0.0
                    results.append({
                        "page": idx + 1,
                        "text": text or "",
                        "confidence": float(avg_conf),
                        "word_count": len((text or "").split())
                    })
                except Exception:
                    results.append({"page": idx + 1, "text": "", "confidence": 0.0, "word_count": 0})

            doc.close()
            return {"success": True, "pages": results, "total_pages": total_pages, "method": "ocr"}
        except Exception as e:
            try:
                doc.close()
            except Exception:
                pass
            return {"success": False, "error": str(e), "method": "ocr"}

    def hybrid_extraction(self, pdf_path: str, min_text_len: int = 50) -> Dict[str, Any]:
        """Try native text extraction first; fall back to OCR for weak pages.

        Returns dict: { success, pages: [ {page, text, confidence, word_count} ], total_pages, method }
        """
        from PyPDF2 import PdfReader
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
        except Exception:
            return self.extract_text_with_ocr(pdf_path)

        results: List[Dict[str, Any]] = []
        need_ocr_indices: List[int] = []
        for i in range(total_pages):
            try:
                txt = reader.pages[i].extract_text() or ""
                wc = len(txt.split())
                if len(txt.strip()) >= min_text_len:
                    results.append({"page": i + 1, "text": txt, "confidence": 95.0, "word_count": wc})
                else:
                    results.append(None)  # type: ignore
                    need_ocr_indices.append(i)
            except Exception:
                results.append(None)  # type: ignore
                need_ocr_indices.append(i)

        if need_ocr_indices:
            ocr = self.extract_text_with_ocr(pdf_path, pages=[idx + 1 for idx in need_ocr_indices])
            if ocr.get("success"):
                ocr_pages_map = {p["page"] - 1: p for p in ocr.get("pages", [])}
                for idx in need_ocr_indices:
                    p = ocr_pages_map.get(idx, {})
                    results[idx] = {
                        "page": idx + 1,
                        "text": p.get("text", ""),
                        "confidence": float(p.get("confidence", 0.0)),
                        "word_count": int(p.get("word_count", 0))
                    }
            else:
                for idx in need_ocr_indices:
                    results[idx] = {"page": idx + 1, "text": "", "confidence": 0.0, "word_count": 0}

        results = [r if isinstance(r, dict) else {"page": i + 1, "text": "", "confidence": 0.0, "word_count": 0} for i, r in enumerate(results)]
        return {"success": True, "pages": results, "total_pages": total_pages, "method": "hybrid"}

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        try:
            gray = ImageOps.grayscale(img)
            blurred = gray.filter(ImageFilter.MedianFilter(size=3))
            enhanced = ImageOps.autocontrast(blurred)
            bw = enhanced.point(lambda x: 255 if x > 160 else 0, mode='1')
            return bw
        except Exception:
            return img

    def _is_int(self, v: Any) -> bool:
        try:
            int(v)
            return True
        except Exception:
            return False
