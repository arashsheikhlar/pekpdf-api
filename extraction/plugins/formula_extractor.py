# SPDX-License-Identifier: AGPL-3.0-only

"""
Formula extraction plugin.

This plugin extracts mathematical formulas and equations from PDF documents.
Currently a placeholder implementation for future formula extraction capabilities.
"""

import re
from typing import Any, Dict, List

from .base import ExtractorPlugin
from ..models import ExtractionOptions, PluginMetadata


class FormulaExtractorPlugin(ExtractorPlugin):
    """Plugin for extracting mathematical formulas from PDF documents."""
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="formula_extractor",
            version="1.0.0",
            description="Extract mathematical formulas and equations from PDF documents",
            supported_types=["research", "financial", "general"],
            priority=40
        )
    
    def can_handle(self, document_type: str, options: ExtractionOptions) -> bool:
        """
        Check if this plugin can handle formula extraction.
        
        Args:
            document_type: Type of document
            options: Extraction options
            
        Returns:
            True if formula extraction is requested or document type supports formulas
        """
        # Check if force_formulas is enabled (would be in custom_instructions)
        if "force_formulas" in options.custom_instructions.lower():
            return True
        
        # Check if document type typically contains formulas
        formula_document_types = ["research", "financial"]
        return document_type in formula_document_types
    
    def extract(self, text: str, pages_text: List[str], options: ExtractionOptions) -> Dict[str, Any]:
        """
        Extract formulas from the document.
        
        Args:
            text: Combined text from all pages
            pages_text: List of text for each page
            options: Extraction options
            
        Returns:
            Dictionary with extracted formulas
        """
        formulas = []
        
        # Extract formulas from each page
        for page_num, page_text in enumerate(pages_text):
            if page_text.strip():
                page_formulas = self._extract_formulas_from_page(page_text, page_num + 1)
                formulas.extend(page_formulas)
        
        return {
            "formulas": formulas,
            "formula_count": len(formulas),
            "extraction_method": "regex_patterns",
            "plugin": "formula_extractor"
        }
    
    def _extract_formulas_from_page(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract formulas from a specific page.
        
        Args:
            page_text: Text content of the page
            page_num: Page number (1-indexed)
            
        Returns:
            List of formulas found on the page
        """
        formulas = []
        
        # Pattern 1: LaTeX-style formulas (between $ or $$)
        latex_patterns = self._extract_latex_formulas(page_text, page_num)
        formulas.extend(latex_patterns)
        
        # Pattern 2: Mathematical expressions with common symbols
        math_patterns = self._extract_math_expressions(page_text, page_num)
        formulas.extend(math_patterns)
        
        # Pattern 3: Equation-like structures
        equation_patterns = self._extract_equations(page_text, page_num)
        formulas.extend(equation_patterns)
        
        # Pattern 4: Fraction-like structures
        fraction_patterns = self._extract_fractions(page_text, page_num)
        formulas.extend(fraction_patterns)
        
        return formulas
    
    def _extract_latex_formulas(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract LaTeX-style formulas."""
        formulas = []
        
        # Single dollar signs (inline math)
        inline_pattern = r'\$([^$]+)\$'
        matches = re.finditer(inline_pattern, text)
        for match in matches:
            formulas.append({
                "type": "latex_inline",
                "content": match.group(1),
                "page": page_num,
                "position": match.start(),
                "format": "latex"
            })
        
        # Double dollar signs (display math)
        display_pattern = r'\$\$([^$]+)\$\$'
        matches = re.finditer(display_pattern, text)
        for match in matches:
            formulas.append({
                "type": "latex_display",
                "content": match.group(1),
                "page": page_num,
                "position": match.start(),
                "format": "latex"
            })
        
        return formulas
    
    def _extract_math_expressions(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract mathematical expressions with common symbols."""
        formulas = []
        
        # Pattern for expressions with mathematical operators
        math_pattern = r'[a-zA-Z0-9\s]+\s*[+\-*/=<>≤≥≠±]\s*[a-zA-Z0-9\s]+'
        matches = re.finditer(math_pattern, text)
        for match in matches:
            content = match.group(0).strip()
            # Filter out simple comparisons and basic arithmetic
            if len(content) > 10 and any(symbol in content for symbol in ['=', '+', '-', '*', '/', '^', '√']):
                formulas.append({
                    "type": "math_expression",
                    "content": content,
                    "page": page_num,
                    "position": match.start(),
                    "format": "text"
                })
        
        return formulas
    
    def _extract_equations(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract equation-like structures."""
        formulas = []
        
        # Pattern for equations (contains = and mathematical symbols)
        equation_pattern = r'[a-zA-Z0-9\s\(\)\[\]+\-*/=<>≤≥≠±^√]+=+[a-zA-Z0-9\s\(\)\[\]+\-*/=<>≤≥≠±^√]+'
        matches = re.finditer(equation_pattern, text)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 15:  # Filter out very short matches
                formulas.append({
                    "type": "equation",
                    "content": content,
                    "page": page_num,
                    "position": match.start(),
                    "format": "text"
                })
        
        return formulas
    
    def _extract_fractions(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract fraction-like structures."""
        formulas = []
        
        # Pattern for fractions (a/b, a over b, etc.)
        fraction_patterns = [
            r'\b[a-zA-Z0-9]+\s*/\s*[a-zA-Z0-9]+\b',  # a/b
            r'\b[a-zA-Z0-9]+\s+over\s+[a-zA-Z0-9]+\b',  # a over b
            r'\b[a-zA-Z0-9]+\s+÷\s+[a-zA-Z0-9]+\b',  # a ÷ b
        ]
        
        for pattern in fraction_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                content = match.group(0).strip()
                formulas.append({
                    "type": "fraction",
                    "content": content,
                    "page": page_num,
                    "position": match.start(),
                    "format": "text"
                })
        
        return formulas
    
    def _detect_mathematical_symbols(self, text: str) -> List[str]:
        """
        Detect mathematical symbols in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of mathematical symbols found
        """
        math_symbols = [
            '∑', '∏', '∫', '∂', '∇', '∞', '±', '∓', '×', '÷', '√', '∛', '∜',
            '≤', '≥', '≠', '≈', '≡', '∝', '∈', '∉', '⊂', '⊃', '∪', '∩',
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'λ', 'μ', 'π', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',
            'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω'
        ]
        
        found_symbols = []
        for symbol in math_symbols:
            if symbol in text:
                found_symbols.append(symbol)
        
        return found_symbols
    
    def _classify_formula_complexity(self, formula: str) -> str:
        """
        Classify formula complexity.
        
        Args:
            formula: Formula content
            
        Returns:
            Complexity level: simple, medium, complex
        """
        # Count mathematical symbols
        symbol_count = len(re.findall(r'[+\-*/=<>≤≥≠±^√∑∏∫∂∇∞]', formula))
        
        # Count variables (single letters)
        variable_count = len(re.findall(r'\b[a-zA-Z]\b', formula))
        
        # Count parentheses
        paren_count = formula.count('(') + formula.count(')')
        
        complexity_score = symbol_count + variable_count + paren_count
        
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 8:
            return "medium"
        else:
            return "complex"
