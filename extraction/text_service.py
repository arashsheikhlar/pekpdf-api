# SPDX-License-Identifier: AGPL-3.0-only

"""
Text extraction service for PDF documents.

This module handles text extraction from PDF files, including both native text extraction
and OCR fallback for scanned documents.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from PyPDF2 import PdfReader

from .models import ExtractionOptions
from .ocr_service import OCRService


class TextExtractionService:
    """Service for extracting text from PDF documents."""
    
    def __init__(self, ocr_service: Optional[OCRService] = None, max_workers: int = 4):
        """
        Initialize the text extraction service.
        
        Args:
            ocr_service: Optional OCR service for scanned documents
            max_workers: Maximum number of worker threads for parallel extraction
        """
        self.ocr_service = ocr_service
        self.max_workers = max_workers
    
    def extract_text_from_pdf(self, file_path: str, use_ocr: bool = False) -> Tuple[str, List[str]]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            use_ocr: Whether to force OCR instead of native text extraction
            
        Returns:
            Tuple of (combined_text, pages_text) where:
            - combined_text: All text joined together
            - pages_text: List of text for each page
        """
        if use_ocr and self.ocr_service:
            return self._extract_with_ocr(file_path)
        else:
            return self._extract_native_text(file_path)
    
    def _extract_native_text(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Extract text using native PDF text extraction.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (combined_text, pages_text)
        """
        try:
            reader = PdfReader(file_path)
            pages_text = []
            
            # Try parallel extraction first
            try:
                def _extract_page(i):
                    try:
                        return reader.pages[i].extract_text() or ""
                    except Exception:
                        return ""
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    pages_text = list(executor.map(_extract_page, range(len(reader.pages))))
            except Exception:
                # Fallback to sequential extraction
                pages_text = []
                for page in reader.pages:
                    try:
                        pages_text.append(page.extract_text() or "")
                    except Exception:
                        pages_text.append("")
            
            all_text = "\n".join(pages_text)
            return all_text, pages_text
            
        except Exception as e:
            # If native extraction fails and OCR is available, try OCR
            if self.ocr_service:
                return self._extract_with_ocr(file_path)
            else:
                # Return empty results if both fail
                return "", []
    
    def _extract_with_ocr(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Extract text using OCR service.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (combined_text, pages_text)
        """
        if not self.ocr_service:
            return "", []
        
        try:
            # Use OCR service to extract text
            result = self.ocr_service.extract_text(file_path)
            if result and isinstance(result, dict):
                pages_text = result.get('pages', [])
                all_text = result.get('text', '')
                return all_text, pages_text
            else:
                return "", []
        except Exception:
            return "", []
    
    def build_inverted_index(self, pages_text: List[str]) -> Dict[str, List[int]]:
        """
        Build a simple inverted index from page text.
        
        Creates a mapping of words (>=4 chars) to sorted list of page numbers (1-indexed).
        
        Args:
            pages_text: List of text content for each page
            
        Returns:
            Dictionary mapping words to lists of page numbers
        """
        index: Dict[str, set] = {}
        try:
            for i, txt in enumerate(pages_text or []):
                if not txt:
                    continue
                # Extract words with 4+ characters
                words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", txt.lower())
                for word in words:
                    if word not in index:
                        index[word] = set()
                    index[word].add(i + 1)  # 1-indexed page numbers
            
            # Convert sets to sorted lists
            return {word: sorted(list(pages)) for word, pages in index.items()}
        except Exception:
            return {}
    
    def get_text_statistics(self, pages_text: List[str]) -> Dict[str, int]:
        """
        Get basic statistics about the extracted text.
        
        Args:
            pages_text: List of text content for each page
            
        Returns:
            Dictionary with text statistics
        """
        try:
            total_chars = sum(len(page) for page in pages_text)
            total_words = sum(len(re.findall(r'\b\w+\b', page)) for page in pages_text)
            non_empty_pages = sum(1 for page in pages_text if page.strip())
            
            return {
                'total_pages': len(pages_text),
                'non_empty_pages': non_empty_pages,
                'total_characters': total_chars,
                'total_words': total_words,
                'avg_chars_per_page': total_chars // len(pages_text) if pages_text else 0,
                'avg_words_per_page': total_words // len(pages_text) if pages_text else 0
            }
        except Exception:
            return {
                'total_pages': 0,
                'non_empty_pages': 0,
                'total_characters': 0,
                'total_words': 0,
                'avg_chars_per_page': 0,
                'avg_words_per_page': 0
            }
    
    def detect_text_quality(self, pages_text: List[str]) -> Dict[str, any]:
        """
        Detect text quality indicators.
        
        Args:
            pages_text: List of text content for each page
            
        Returns:
            Dictionary with quality indicators
        """
        try:
            total_chars = sum(len(page) for page in pages_text)
            total_words = sum(len(re.findall(r'\b\w+\b', page)) for page in pages_text)
            
            # Check for OCR-like patterns (common OCR artifacts)
            ocr_indicators = 0
            for page in pages_text:
                # Check for common OCR mistakes
                if re.search(r'[0O]', page):  # 0/O confusion
                    ocr_indicators += 1
                if re.search(r'[Il1]', page):  # I/l/1 confusion
                    ocr_indicators += 1
                if re.search(r'[rn]', page):  # rn confusion
                    ocr_indicators += 1
            
            # Calculate quality score
            quality_score = max(0, 100 - (ocr_indicators * 10))
            
            return {
                'quality_score': quality_score,
                'likely_ocr': ocr_indicators > len(pages_text) * 0.3,
                'ocr_indicators': ocr_indicators,
                'text_density': total_chars / len(pages_text) if pages_text else 0,
                'word_density': total_words / len(pages_text) if pages_text else 0
            }
        except Exception:
            return {
                'quality_score': 0,
                'likely_ocr': False,
                'ocr_indicators': 0,
                'text_density': 0,
                'word_density': 0
            }
