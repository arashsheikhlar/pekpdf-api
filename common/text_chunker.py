"""
Text chunking utilities for handling large documents.
"""
import re
from typing import List, Dict, Tuple


def chunk_text_with_overlap(text: str, max_chars: int = 12000, overlap: int = 500) -> List[Dict[str, any]]:
    """
    Split text into chunks with overlap for context preservation.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunks with metadata: [{"text": str, "start": int, "end": int}, ...]
    """
    if len(text) <= max_chars:
        return [{"text": text, "start": 0, "end": len(text), "chunk_id": 0}]
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + max_chars
        
        # If not at the end, try to break at a sentence or paragraph
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + max_chars // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break > start + max_chars // 2:
                    end = sentence_break + 2
        
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "start": start,
            "end": end,
            "chunk_id": chunk_id
        })
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else end
        chunk_id += 1
    
    return chunks


def find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Find most relevant chunks for a query using simple keyword similarity.
    
    Args:
        query: Search query
        chunks: List of chunk dictionaries
        top_k: Number of top chunks to return
    
    Returns:
        Top k most relevant chunks
    """
    if not chunks or not query:
        return chunks[:top_k] if chunks else []
    
    query_words = set(query.lower().split())
    
    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        chunk_text = chunk["text"].lower()
        chunk_words = set(chunk_text.split())
        
        # Simple scoring: count matching words
        matches = len(query_words & chunk_words)
        
        # Bonus for exact phrase match
        if query.lower() in chunk_text:
            matches += 10
        
        scored_chunks.append({
            **chunk,
            "relevance_score": matches
        })
    
    # Sort by relevance score
    scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return scored_chunks[:top_k]


def extract_page_numbers(text: str) -> List[int]:
    """
    Extract page number references from text.
    Looks for patterns like "page 5", "p. 10", "[Page 3]", etc.
    
    Args:
        text: Text to extract page numbers from
    
    Returns:
        List of page numbers found
    """
    page_numbers = set()
    
    # Pattern 1: [Page X]
    matches = re.findall(r'\[Page\s+(\d+)\]', text, re.IGNORECASE)
    page_numbers.update(int(m) for m in matches)
    
    # Pattern 2: page X, p. X, pg. X
    matches = re.findall(r'\b(?:page|p\.|pg\.)\s*(\d+)\b', text, re.IGNORECASE)
    page_numbers.update(int(m) for m in matches)
    
    # Pattern 3: (p.X)
    matches = re.findall(r'\(p\.?\s*(\d+)\)', text, re.IGNORECASE)
    page_numbers.update(int(m) for m in matches)
    
    return sorted(list(page_numbers))


def extract_text_with_page_markers(text: str) -> List[Tuple[int, str]]:
    """
    Split text by page markers and return list of (page_num, text) tuples.
    
    Args:
        text: Text with [Page X] markers
    
    Returns:
        List of (page_number, page_text) tuples
    """
    pages = []
    parts = re.split(r'\[Page\s+(\d+)\]', text, flags=re.IGNORECASE)
    
    # First part is before any page marker
    if parts[0].strip():
        pages.append((0, parts[0].strip()))
    
    # Remaining parts alternate between page numbers and content
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_num = int(parts[i])
            page_text = parts[i + 1].strip()
            if page_text:
                pages.append((page_num, page_text))
    
    return pages

