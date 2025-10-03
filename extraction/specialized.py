# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, Dict, List, Tuple
import os
import re

class TableExtractor:
    """Extract simple tables from PDFs using pdfplumber (best-effort).
    Returns a list of tables, each { page, rows: List[List[str]] }.
    """

    def extract(self, pdf_path: str, max_pages: int = 50, max_tables: int = 20, page_numbers: List[int] | None = None) -> List[Dict[str, Any]]:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return []
        if not os.path.isfile(pdf_path):
            return []
        tables: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                # page_numbers are 1-indexed; clamp to available range
                if isinstance(page_numbers, list) and page_numbers:
                    candidates = [p for p in sorted(set(page_numbers)) if 1 <= p <= total_pages]
                else:
                    candidates = list(range(1, min(total_pages, max_pages) + 1))
                for one_index in candidates:
                    if one_index > max_pages and page_numbers is None:
                        break
                    page_index = one_index - 1
                    page = pdf.pages[page_index]
                    try:
                        page_tables = page.extract_tables() or []
                    except Exception:
                        page_tables = []
                    for tbl in page_tables:
                        if tbl:
                            rows = []
                            for row in tbl:
                                rows.append(["" if cell is None else str(cell).strip() for cell in row])
                            tables.append({"page": one_index, "rows": rows})
                            if len(tables) >= max_tables:
                                return tables
        except Exception:
            pass
        return tables


class FormulaExtractor:
    """Extract LaTeX-like math expressions from text, including inline and display math.
    Returns a list of unique formula strings.
    """

    # Matches $...$, $$...$$, \(...\), \[...\]
    INLINE_DOLLAR = re.compile(r"\$(.+?)\$", re.DOTALL)
    DISPLAY_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    INLINE_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
    DISPLAY_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

    # Heuristic LaTeX command presence
    TEX_CMD = re.compile(r"\\(?:frac|sum|int|alpha|beta|gamma|theta|mu|sigma|pm|cdot|times|leq|geq|approx|begin\{|end\{|vec|hat|bar)")

    def extract(self, text: str, max_formulas: int = 200) -> List[str]:
        if not text:
            return []
        found: List[str] = []
        def _cap(matches):
            for m in matches:
                s = m.strip()
                if s:
                    found.append(s)
        _cap(self.DISPLAY_DOLLAR.findall(text))
        _cap(self.DISPLAY_BRACKET.findall(text))
        _cap(self.INLINE_PAREN.findall(text))
        _cap(self.INLINE_DOLLAR.findall(text))

        # Add lines with LaTeX commands even if not in math fences
        lines = [ln.strip() for ln in text.split("\n")]
        for ln in lines:
            if len(found) >= max_formulas:
                break
            if self.TEX_CMD.search(ln):
                found.append(ln)

        # Deduplicate, preserve order
        seen = set()
        ordered = []
        for f in found:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
            if len(ordered) >= max_formulas:
                break
        return ordered


class Router:
    """Decide which specialized extractors to run based on doc type and hints."""

    def decide(self, dtype: str, text: str, force_tables: bool | None = None, force_formulas: bool | None = None) -> Tuple[bool, bool]:
        dtype = (dtype or 'general').lower()
        need_tables = False
        need_formulas = False
        if force_tables is not None:
            need_tables = bool(force_tables)
        if force_formulas is not None:
            need_formulas = bool(force_formulas)
        if force_tables is None:
            if dtype in ("financial", "research"):
                # heuristic: mention of Table or tabular markers
                if re.search(r"\b(Table|Tab\.)\s*\d+", text, flags=re.IGNORECASE) or ("%" in text):
                    need_tables = True
        if force_formulas is None:
            if dtype == "research":
                if re.search(r"\$|\\\(|\\\[|\\frac|\\sum|\\int|\\beta|\\alpha", text):
                    need_formulas = True
        return need_tables, need_formulas 