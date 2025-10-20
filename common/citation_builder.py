"""
Citation building utilities for mapping AI responses to source pages.
"""
import re
from typing import List, Dict, Tuple


def build_citations(ai_response: str, pages_data: List[Dict]) -> List[Dict]:
    """
    Map AI response statements to source pages using keyword matching.
    
    Args:
        ai_response: AI-generated response text
        pages_data: List of dicts with 'page' and 'text' keys
    
    Returns:
        List of citations: [{"text": str, "pages": [int], "confidence": float}, ...]
    """
    if not ai_response or not pages_data:
        return []
    
    citations = []
    
    # Split response into sentences for citation mapping
    sentences = split_into_sentences(ai_response)
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:  # Skip very short sentences
            continue
        
        # Find which pages contain similar content
        relevant_pages = find_source_pages(sentence, pages_data)
        
        if relevant_pages:
            citations.append({
                "text": sentence.strip(),
                "pages": relevant_pages,
                "confidence": calculate_confidence(sentence, pages_data, relevant_pages)
            })
    
    # Remove duplicates and consolidate
    citations = consolidate_citations(citations)
    
    return citations


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def find_source_pages(sentence: str, pages_data: List[Dict], min_match_words: int = 3) -> List[int]:
    """
    Find which pages contain content related to a sentence.
    
    Args:
        sentence: Sentence to find sources for
        pages_data: List of page dictionaries
        min_match_words: Minimum number of matching words to consider a page relevant
    
    Returns:
        List of page numbers
    """
    # Extract meaningful words (longer than 3 chars, not common words)
    stop_words = {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from', 'have', 'has', 'was', 'were', 'been', 'their', 'they', 'them', 'what', 'when', 'where', 'which', 'who', 'will', 'would', 'could', 'should'}
    words = [w.lower() for w in re.findall(r'\b\w{4,}\b', sentence) if w.lower() not in stop_words]
    
    if not words:
        return []
    
    relevant_pages = []
    
    for page_data in pages_data:
        page_text = page_data.get('text', '').lower()
        
        # Count how many sentence words appear in this page
        matches = sum(1 for word in words if word in page_text)
        
        # Check for phrase matches (higher value)
        phrase_matches = 0
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if phrase in page_text:
                phrase_matches += 1
        
        total_score = matches + (phrase_matches * 2)
        
        if total_score >= min_match_words:
            relevant_pages.append(page_data['page'])
    
    # Limit to top 3 pages
    return relevant_pages[:3]


def calculate_confidence(sentence: str, pages_data: List[Dict], relevant_pages: List[int]) -> float:
    """
    Calculate confidence score for a citation.
    
    Args:
        sentence: Sentence being cited
        pages_data: All page data
        relevant_pages: Pages identified as sources
    
    Returns:
        Confidence score between 0 and 1
    """
    if not relevant_pages:
        return 0.0
    
    # Base confidence
    confidence = 0.5
    
    # Boost if multiple pages agree
    if len(relevant_pages) >= 2:
        confidence += 0.2
    
    # Boost if exact phrases found
    words = re.findall(r'\b\w{4,}\b', sentence.lower())
    for page_num in relevant_pages:
        page_data = next((p for p in pages_data if p['page'] == page_num), None)
        if page_data:
            page_text = page_data['text'].lower()
            # Check for 4+ word sequences
            for i in range(len(words) - 3):
                phrase = ' '.join(words[i:i+4])
                if phrase in page_text:
                    confidence += 0.1
                    break
    
    return min(confidence, 1.0)


def consolidate_citations(citations: List[Dict]) -> List[Dict]:
    """
    Remove duplicate citations and consolidate similar ones.
    
    Args:
        citations: List of citation dicts
    
    Returns:
        Consolidated list of citations
    """
    if not citations:
        return []
    
    # Remove exact duplicates
    seen = set()
    unique_citations = []
    
    for citation in citations:
        key = (citation['text'], tuple(citation['pages']))
        if key not in seen:
            seen.add(key)
            unique_citations.append(citation)
    
    return unique_citations


def format_response_with_citations(response: str, citations: List[Dict]) -> str:
    """
    Format response text with inline citation markers [p.X].
    
    Args:
        response: Original AI response
        citations: List of citations
    
    Returns:
        Response with inline citations
    """
    if not citations:
        return response
    
    formatted = response
    
    # Add citations after sentences
    for citation in citations:
        sentence = citation['text']
        pages = citation['pages']
        
        if pages and sentence in formatted:
            # Create citation marker
            if len(pages) == 1:
                marker = f" [p.{pages[0]}]"
            elif len(pages) <= 3:
                marker = f" [p.{', '.join(map(str, pages))}]"
            else:
                marker = f" [p.{pages[0]}-{pages[-1]}]"
            
            # Add marker after the sentence (only once)
            formatted = formatted.replace(sentence, sentence + marker, 1)
    
    return formatted


def extract_citations_from_response(response: str) -> List[Dict]:
    """
    Extract existing citation markers from a response.
    
    Args:
        response: Response text with citation markers
    
    Returns:
        List of citations found
    """
    citations = []
    
    # Pattern: [p.X] or [p.X, Y, Z] or [p.X-Y]
    pattern = r'\[p\.(\d+(?:[-,]\s*\d+)*)\]'
    
    matches = re.finditer(pattern, response)
    for match in matches:
        pages_str = match.group(1)
        
        # Parse page numbers
        if '-' in pages_str:
            # Range like "5-8"
            start, end = map(int, pages_str.split('-'))
            pages = list(range(start, end + 1))
        elif ',' in pages_str:
            # List like "1, 3, 5"
            pages = [int(p.strip()) for p in pages_str.split(',')]
        else:
            # Single page
            pages = [int(pages_str)]
        
        citations.append({
            "text": match.group(0),
            "pages": pages,
            "position": match.start()
        })
    
    return citations

