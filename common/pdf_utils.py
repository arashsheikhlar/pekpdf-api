"""
PDF utility functions for text extraction, density checks, and page handling.
"""
import os
import PyPDF2
from typing import List, Dict, Tuple


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_text_by_page(pdf_path: str) -> List[str]:
    """Extract text from each page separately."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text())
            return pages
    except Exception as e:
        raise Exception(f"Failed to extract pages from PDF: {str(e)}")


def compute_text_density(pdf_path: str) -> float:
    """
    Compute text density as chars per page.
    Low density suggests scanned/image-based PDF requiring OCR.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
        
        if page_count == 0:
            return 0.0
        
        char_count = len(text.strip())
        return char_count / page_count
    except:
        return 0.0


def get_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except:
        return 0

