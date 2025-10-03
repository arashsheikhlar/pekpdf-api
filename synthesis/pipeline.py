from typing import List, Tuple, Callable, Dict, Any
from PyPDF2 import PdfReader
import re
import os


class DocumentAnalyzer:
    """Extracts text snippets from PDFs with per-file and global caps, and counts pages."""

    def analyze(self, files, per_file_max_chars: int, global_max_chars: int) -> Tuple[List[Dict[str, Any]], List[str], int]:
        documents: List[Dict[str, Any]] = []
        combined_snippets: List[str] = []
        total_pages: int = 0
        current_total_chars: int = 0

        for file_index, f in enumerate(files):
            if current_total_chars >= global_max_chars:
                break
            file_name = getattr(f, 'filename', 'unknown.pdf') or 'unknown.pdf'
            try:
                reader = PdfReader(f)
                page_texts: List[str] = []
                doc_text = ""
                for p in reader.pages:
                    txt = p.extract_text() or ""
                    page_texts.append(txt)
                    if txt:
                        doc_text += txt + "\n"
                pages_count = len(page_texts)
                total_pages += pages_count

                snippet = (doc_text or "").strip()
                if snippet:
                    snippet = snippet[:per_file_max_chars]
                    remaining = max(global_max_chars - current_total_chars, 0)
                    if remaining > 0:
                        snippet = snippet[:remaining]
                        if snippet:
                            combined_snippets.append(snippet)
                            current_total_chars += len(snippet)
                else:
                    combined_snippets.append(f"[Unreadable or no text extracted from {file_name}]")

                documents.append({
                    "file_index": file_index,
                    "filename": file_name,
                    "pages_count": pages_count,
                    "page_texts": page_texts,
                    "snippet": snippet or "",
                })
            except Exception:
                documents.append({
                    "file_index": file_index,
                    "filename": file_name,
                    "pages_count": 0,
                    "page_texts": [],
                    "snippet": "",
                    "error": f"Could not read {file_name}",
                })
                combined_snippets.append(f"[Unreadable or no text extracted from {file_name}]")

        if not combined_snippets:
            combined_snippets.append("[No text extracted from any input documents]")

        return documents, combined_snippets, total_pages

    # Backwards-compatible method if needed elsewhere
    def extract_text_snippets(self, files, per_file_max_chars: int, global_max_chars: int) -> Tuple[List[str], int]:
        docs, snippets, total_pages = self.analyze(files, per_file_max_chars, global_max_chars)
        return snippets, total_pages


class DocumentClassifier:
    """Very lightweight heuristic classifier to tag documents by domain and topics."""

    KEYWORDS = {
        "legal": ["plaintiff", "defendant", "contract", "jurisdiction", "statute", "regulation", "holding", "motion"],
        "finance": ["revenue", "ebitda", "margin", "cash flow", "balance sheet", "income statement", "forecast", "kpi"],
        "research": ["methodology", "dataset", "experiment", "hypothesis", "statistical", "significance", "results", "findings"],
        "healthcare": ["patient", "clinical", "diagnosis", "treatment", "protocol", "outcomes", "contraindication", "dose"],
    }

    def classify(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for doc in documents:
            text = (doc.get("snippet") or "").lower()
            domain_scores: Dict[str, int] = {}
            for domain, kws in self.KEYWORDS.items():
                score = sum(1 for kw in kws if kw in text)
                if score:
                    domain_scores[domain] = score
            top_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
            doc_enriched = {**doc, "domain": top_domain, "domain_scores": domain_scores}
            enriched.append(doc_enriched)
        return enriched


class ConflictDetector:
    """Detects simple numeric conflicts across documents for similarly-labeled facts."""

    NUM_RE = re.compile(r"(?P<value>[-+]?\d[\d,\.]*\s?(?:%|million|billion|bn|k|m)?|\$\s?\d[\d,\.]*)", re.IGNORECASE)

    def _extract_facts(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        # Search in the first N pages to keep it light
        search_text = (doc.get("snippet") or "")
        # Build (label, value) pairs by grabbing a small left context around numbers
        for m in self.NUM_RE.finditer(search_text):
            start = max(0, m.start() - 40)
            context = search_text[start:m.start()].strip()
            label = context.split()[-5:]
            label = " ".join(label).strip().lower()
            value_raw = m.group("value").strip()
            value_norm = value_raw.replace(",", "").lower()
            facts.append({
                "label": label or "value",
                "value": value_norm,
                "raw": value_raw,
                "file_index": doc.get("file_index"),
                "filename": doc.get("filename"),
            })
        return facts

    def detect(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_facts: List[Dict[str, Any]] = []
        for doc in documents:
            all_facts.extend(self._extract_facts(doc))
        # Group by label (coarse) and check for conflicting values across files
        conflicts: List[Dict[str, Any]] = []
        facts_by_label: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for f in all_facts:
            label = f.get("label", "value")
            val = f.get("value", "")
            facts_by_label.setdefault(label, {}).setdefault(val, []).append(f)
        for label, by_val in facts_by_label.items():
            if len(by_val.keys()) > 1:
                entries = []
                for val, facts in by_val.items():
                    entries.append({
                        "value": val,
                        "occurrences": [{"file_index": x["file_index"], "filename": x["filename"], "raw": x["raw"]} for x in facts]
                    })
                conflicts.append({"label": label, "values": entries})
        return conflicts


class ProvenanceMapper:
    """Maps synthesized lines back to likely source files/pages using token overlap."""

    def _score_overlap(self, a: str, b: str) -> float:
        wa = set(w for w in re.findall(r"[a-zA-Z0-9]+", a.lower()) if len(w) > 2)
        wb = set(w for w in re.findall(r"[a-zA-Z0-9]+", b.lower()) if len(w) > 2)
        if not wa or not wb:
            return 0.0
        inter = len(wa & wb)
        union = len(wa | wb)
        return inter / union if union else 0.0

    def map(self, ai_text: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in (ai_text or "").split("\n")]
        lines = [ln for ln in lines if ln]
        mappings: List[Dict[str, Any]] = []
        for idx, ln in enumerate(lines):
            best: List[Tuple[float, int, int]] = []  # (score, file_index, page_number)
            for doc in documents:
                file_idx = doc.get("file_index")
                for page_num, page_text in enumerate(doc.get("page_texts", []), start=1):
                    score = self._score_overlap(ln, page_text)
                    if score > 0:
                        best.append((score, file_idx, page_num))
            best.sort(reverse=True)
            top = best[:5]
            if top:
                mappings.append({
                    "line_index": idx,
                    "line": ln,
                    "sources": [{"file_index": fi, "page": pg, "score": round(sc, 3)} for sc, fi, pg in top]
                })
        return mappings


class AdvancedDocumentAnalyzer:
    """Extract structured information from document snippets to improve downstream synthesis."""

    DATE_PATTERNS = [
        r"\b\d{4}-\d{2}-\d{2}\b",                # 2024-09-30
        r"\b\d{2}/\d{2}/\d{4}\b",                # 09/30/2025
        r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    ]

    ENTITY_HINTS = {
        "organization": [r"\b[A-Z][A-Za-z0-9&.,\-]+\s+(Inc\.|LLC|Ltd\.|Corp\.|PLC|GmbH|S\.A\.|N\.V\.|AG)\b"],
        "statute": [r"\b\d+\s+U\.S\.C\.\s+\d+[a-zA-Z0-9()]*\b", r"\b\d+\s+CFR\s+\d+\.?\d*\b"],
        "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"],
        "case": [r"\b[A-Z][A-Za-z\- ]+\s+v\.\s+[A-Z][A-Za-z\- ]+\b"],
    }

    NUM_RE = re.compile(r"(?P<value>[-+]?\d[\d,\.]*\s?(?:%|million|billion|bn|k|m)?|\$\s?\d[\d,\.]*)", re.IGNORECASE)

    DOC_TYPE_KEYWORDS = {
        "contract": ["agreement", "party", "parties", "term", "termination", "jurisdiction", "governing law", "liability"],
        "invoice": ["invoice", "bill to", "due date", "subtotal", "tax", "balance due"],
        "financial_statement": ["income statement", "balance sheet", "cash flow", "ebitda", "margin", "assets", "liabilities"],
        "research_paper": ["abstract", "introduction", "methodology", "results", "discussion", "conclusion", "references"],
        "medical_report": ["patient", "diagnosis", "treatment", "medication", "dosage", "clinical", "vitals"],
        "minutes": ["meeting", "attendees", "agenda", "action items", "decisions", "next steps"],
        "email": ["from:", "to:", "subject:", "cc:", "sent:"],
    }

    def extract_dates(self, text: str) -> List[str]:
        dates: List[str] = []
        for pat in self.DATE_PATTERNS:
            dates.extend(re.findall(pat, text, flags=re.IGNORECASE))
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for d in dates:
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        return ordered[:200]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {k: [] for k in self.ENTITY_HINTS.keys()}
        # Pattern-based entities
        for kind, patterns in self.ENTITY_HINTS.items():
            found: List[str] = []
            for pat in patterns:
                found.extend(re.findall(pat, text))
            # Normalize and dedupe
            norm = []
            seen = set()
            for f in found:
                s = f if isinstance(f, str) else " ".join(f)
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    norm.append(s)
            entities[kind] = norm[:200]
        # Heuristic organization/person names: consecutive capitalized words (2-6 tokens)
        caps = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+|$)){2,6}", text)
        heur = []
        seen_caps = set()
        for c in caps:
            s = c.strip()
            if len(s.split()) >= 2 and s not in seen_caps and not s.lower().startswith("the "):
                seen_caps.add(s)
                heur.append(s)
        entities.setdefault("proper_nouns", [])
        entities["proper_nouns"] = heur[:200]
        return entities

    def extract_metrics(self, text: str) -> List[Dict[str, str]]:
        metrics: List[Dict[str, str]] = []
        for m in self.NUM_RE.finditer(text):
            start = max(0, m.start() - 50)
            context = text[start:m.start()].strip()
            label = " ".join(context.split()[-6:]).strip().lower()
            value_raw = m.group("value").strip()
            metrics.append({"label": label or "value", "value": value_raw})
        return metrics[:500]

    def detect_document_type(self, text: str) -> str:
        scores: Dict[str, int] = {}
        lower = text.lower()
        for dtype, kws in self.DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in lower)
            if score:
                scores[dtype] = score
        return max(scores, key=scores.get) if scores else "general"

    def extract_sections(self, text: str) -> List[str]:
        lines = [ln.strip() for ln in (text or "").split("\n")]
        headers: List[str] = []
        for ln in lines:
            l = ln.strip()
            if not l:
                continue
            if l.endswith(':') and 2 <= len(l) <= 120:
                headers.append(l[:-1].strip())
                continue
            if re.match(r"^(?:[A-Z0-9][A-Z0-9\s\-/()&]{2,80})$", l):
                headers.append(l)
                continue
            if re.match(r"^(?:\d+\.|[A-Z]\))\s+.+$", l):
                headers.append(l)
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for h in headers:
            if h not in seen:
                seen.add(h)
                ordered.append(h)
        return ordered[:200]

    def enrich(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for doc in documents:
            text = doc.get("snippet") or ""
            entities = self.extract_entities(text)
            dates = self.extract_dates(text)
            metrics = self.extract_metrics(text)
            doc_type = self.detect_document_type(text)
            sections = self.extract_sections(text)
            doc_enriched = {
                **doc,
                "analysis": {
                    "entities": entities,
                    "dates": dates,
                    "metrics": metrics,
                    "doc_type": doc_type,
                    "sections": sections,
                }
            }
            enriched.append(doc_enriched)
        return enriched


class TemplateEngine:
    """Selects and applies domain/format-aware templates to guide synthesis structure and tone."""

    def _dominant_domain(self, documents: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = {}
        for d in documents:
            dom = d.get("domain") or "general"
            counts[dom] = counts.get(dom, 0) + 1
        if not counts:
            return "general"
        return max(counts, key=counts.get)

    def _doc_types(self, documents: List[Dict[str, Any]]) -> List[str]:
        types: Dict[str, int] = {}
        for d in documents:
            t = ((d.get("analysis") or {}).get("doc_type") or "general").lower()
            types[t] = types.get(t, 0) + 1
        ordered = sorted(types.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ordered[:3]] or ["general"]

    def _sections_for(self, domain: str, target_format: str, profile: str | None = None) -> List[str]:
        domain = domain or "general"
        fmt = target_format or "report"
        # Profile-led sections override
        profile_sections: Dict[str, List[str]] = {
            "executive_summary": ["Executive Summary", "Key Points", "Implications", "Risks & Mitigations", "Recommendations"],
            "risk_assessment": ["Executive Summary", "Risk Register", "Likelihood & Impact", "Controls & Gaps", "Mitigations", "Next Steps"],
            "compliance_review": ["Executive Summary", "Scope & Standards", "Findings", "Nonconformities", "Remediation Plan", "Recommendations"],
        }
        if profile and profile in profile_sections:
            return profile_sections[profile]

        base = {
            "report": ["Executive Summary", "Key Findings", "Detailed Analysis", "Risks & Mitigations", "Recommendations"],
            "brief": ["Executive Summary", "Key Points", "Implications", "Recommendations"],
            "minutes": ["Meeting Overview", "Attendees", "Agenda", "Decisions", "Action Items", "Next Steps"],
        }
        legal = {
            "report": ["Executive Summary", "Case Background", "Issues", "Arguments & Analysis", "Precedents", "Risks", "Recommendations"],
            "brief": ["Executive Summary", "Key Issues", "Positions", "Considerations", "Recommendations"],
            "minutes": base["minutes"],
        }
        finance = {
            "report": ["Executive Summary", "Performance Overview", "Financial Metrics", "Risk Assessment", "Forecasts", "Recommendations"],
            "brief": ["Executive Summary", "KPIs", "Highlights", "Risks", "Recommendations"],
            "minutes": base["minutes"],
        }
        research = {
            "report": ["Executive Summary", "Background", "Methodology", "Results", "Discussion", "Limitations", "Future Work"],
            "brief": ["Executive Summary", "Key Findings", "Implications", "Limitations", "Next Steps"],
            "minutes": base["minutes"],
        }
        healthcare = {
            "report": ["Executive Summary", "Patient/Population", "Diagnostics", "Interventions", "Outcomes", "Safety/Compliance", "Recommendations"],
            "brief": ["Executive Summary", "Clinical Highlights", "Risks/Safety", "Recommendations"],
            "minutes": base["minutes"],
        }
        by_domain = {
            "legal": legal,
            "finance": finance,
            "research": research,
            "healthcare": healthcare,
        }
        if domain in by_domain and fmt in by_domain[domain]:
            return by_domain[domain][fmt]
        return base.get(fmt, base["report"])

    def build_template(self, documents: List[Dict[str, Any]], target_format: str, forced_domain: str | None = None, profile: str | None = None) -> Dict[str, Any]:
        domain = (forced_domain or self._dominant_domain(documents))
        types = self._doc_types(documents)
        sections = self._sections_for(domain, target_format, profile)
        style = {
            "tone": "professional, concise, evidence-grounded",
            "voice": "active",
            "bullets": "use '-' dashes for bullets; keep items short",
            "headings": "use the provided section titles as H2; avoid markdown asterisks",
        }
        instructions = [
            "Prefer short, scannable paragraphs (2-3 sentences)",
            "Use data points when available; avoid speculation",
            "Keep recommendations actionable and specific",
        ]
        return {
            "name": f"{domain}_{target_format}_template",
            "domain": domain,
            "doc_types": types,
            "sections": sections,
            "style": style,
            "instructions": instructions,
            "profile": profile or "default",
        }


class QualityPipeline:
    """Computes simple quality signals: outline coverage, provenance coverage, and numeric alignment."""

    NUM_RE = re.compile(r"(?P<value>[-+]?\d[\d,\.]*\s?(?:%|million|billion|bn|k|m)?|\$\s?\d[\d,\.]*)", re.IGNORECASE)

    def _normalize(self, s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"[^a-z0-9%$\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def outline_coverage(self, ai_text: str, template: Dict[str, any]) -> Dict[str, any]:
        headings = [self._normalize(h) for h in (template.get("sections") or [])]
        text_norm = self._normalize(ai_text)
        present = []
        missing = []
        for h in headings:
            if not h:
                continue
            present.append(h) if h in text_norm else missing.append(h)
        total = len(headings) if headings else 0
        covered = len(present)
        pct = (covered / total) if total else 1.0
        return {"total": total, "covered": covered, "coverage": round(pct, 3), "missing": missing[:50]}

    def provenance_coverage(self, provenance: any) -> Dict[str, any]:
        if not provenance or not isinstance(provenance, list):
            return {"lines": 0, "with_sources": 0, "coverage": 0.0, "avg_top_score": 0.0}
        lines = len(provenance)
        with_sources = sum(1 for m in provenance if m.get("sources"))
        top_scores = []
        for m in provenance:
            srcs = m.get("sources") or []
            if srcs:
                top_scores.append(float(srcs[0].get("score", 0)))
        avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
        return {"lines": lines, "with_sources": with_sources, "coverage": round(with_sources / lines, 3) if lines else 0.0, "avg_top_score": round(avg_top, 3)}

    def numeric_alignment(self, ai_text: str, documents: List[Dict[str, any]]) -> Dict[str, any]:
        # Check what fraction of numeric tokens in ai_text appear in any source snippet
        nums = [m.group("value").strip() for m in self.NUM_RE.finditer(ai_text or "")]
        if not nums:
            return {"numbers": 0, "matched": 0, "match_ratio": 1.0}
        source_join = "\n".join([d.get("snippet") or "" for d in documents])
        matched = 0
        for n in nums:
            n_norm = self._normalize(n)
            if n_norm and self._normalize(source_join).find(n_norm) >= 0:
                matched += 1
        ratio = matched / len(nums) if nums else 1.0
        return {"numbers": len(nums), "matched": matched, "match_ratio": round(ratio, 3)}

    def evaluate(self, ai_text: str, documents: List[Dict[str, any]], template: Dict[str, any], provenance: any) -> Dict[str, any]:
        outline = self.outline_coverage(ai_text, template or {})
        prov = self.provenance_coverage(provenance)
        numeric = self.numeric_alignment(ai_text, documents)
        # Simple overall score (weighted)
        overall = 0.5 * outline.get("coverage", 0) + 0.3 * prov.get("coverage", 0) + 0.2 * numeric.get("match_ratio", 0)
        return {
            "outline": outline,
            "provenance": prov,
            "numeric": numeric,
            "overall_score": round(overall, 3)
        }


class PromptBuilder:
    """Builds a synthesis prompt from content snippets and selected target format."""

    SYSTEM_MAP = {
        "report": "Produce a well-structured professional report with sections (Overview, Key Findings, Analysis, Recommendations).",
        "brief": "Produce an executive brief with bullets, focusing on clarity and concision.",
        "minutes": "Produce meeting minutes with attendees (if inferable), agenda, decisions, action items, and next steps.",
    }

    def build(self, combined_snippets: List[str], total_pages: int, target_format: str, template: Dict[str, Any] | None = None, user_instructions: str | None = None) -> str:
        system_instruction = self.SYSTEM_MAP.get(target_format, self.SYSTEM_MAP["report"])  # default to report
        outline = "\n".join([f"- {s}" for s in (template.get("sections") if template else [])]) if template else ""
        style = template.get("style") if template else {}
        style_lines = []
        if style:
            if style.get("tone"): style_lines.append(f"Tone: {style['tone']}")
            if style.get("voice"): style_lines.append(f"Voice: {style['voice']}")
            if style.get("bullets"): style_lines.append(f"Bullets: {style['bullets']}")
            if style.get("headings"): style_lines.append(f"Headings: {style['headings']}")
        extra_instructions = template.get("instructions") if template else []
        if user_instructions and user_instructions.strip():
            extra_instructions = list(extra_instructions) + [user_instructions.strip()]
        extra_text = "\n".join(f"- {i}" for i in extra_instructions)

        prompt = (
            f"SYSTEM: You synthesize multiple documents. Return a cohesive {target_format} in well-formatted paragraphs and bullet points where appropriate.\n"
            f"Use clear headings (no markdown asterisks). Prefer headings that end with a colon.\n"
            f"When listing, put each item on its own new line starting with '-' (dash). Avoid using '*' or '**'.\n\n"
            f"DOCUMENT COUNT: {len(combined_snippets)}, TOTAL PAGES (approx): {total_pages}\n"
            f"FORMAT STYLE: {system_instruction}\n\n"
            f"DESIRED OUTLINE (use as section headings where appropriate):\n{outline}\n\n"
            f"STYLE NOTES:\n{os.linesep.join(style_lines)}\n\n"
            f"ADDITIONAL INSTRUCTIONS:\n{extra_text}\n\n"
            f"CONTENT SNIPPETS (truncated):\n"
            + "\n\n---\n\n".join(combined_snippets)
        )
        return prompt


class SynthesisOrchestrator:
    """Coordinates analysis, classification, prompt generation, conflict detection and provenance mapping."""

    def __init__(self,
                 analyzer: DocumentAnalyzer | None = None,
                 classifier: DocumentClassifier | None = None,
                 conflict_detector: ConflictDetector | None = None,
                 provenance_mapper: ProvenanceMapper | None = None,
                 prompt_builder: PromptBuilder | None = None,
                 advanced_analyzer: AdvancedDocumentAnalyzer | None = None,
                 template_engine: TemplateEngine | None = None,
                 quality_pipeline: QualityPipeline | None = None):
        self.analyzer = analyzer or DocumentAnalyzer()
        self.classifier = classifier or DocumentClassifier()
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.provenance_mapper = provenance_mapper or ProvenanceMapper()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.advanced_analyzer = advanced_analyzer or AdvancedDocumentAnalyzer()
        self.template_engine = template_engine or TemplateEngine()
        self.quality_pipeline = quality_pipeline or QualityPipeline()

    def run(self,
            files,
            target_format: str,
            per_file_max_chars: int,
            global_max_chars: int,
            ai_caller: Callable[[str], str],
            forced_domain: str | None = None,
            template_profile: str | None = None,
            user_instructions: str | None = None) -> tuple[str, Dict[str, Any]]:
        documents, combined_snippets, total_pages = self.analyzer.analyze(
            files=files,
            per_file_max_chars=per_file_max_chars,
            global_max_chars=global_max_chars,
        )
        documents = self.classifier.classify(documents)
        documents = self.advanced_analyzer.enrich(documents)

        # Choose template based on analysis and optional overrides
        template = self.template_engine.build_template(documents, target_format, forced_domain=forced_domain, profile=template_profile)

        # Build prompt with template guidance and optional user instructions
        prompt = self.prompt_builder.build(combined_snippets, total_pages, target_format, template, user_instructions=user_instructions)

        ai_text = ai_caller(prompt)
        if not isinstance(ai_text, str):
            ai_text = str(ai_text)
        if not ai_text.strip():
            ai_text = "No synthesized content produced by AI."

        provenance = self.provenance_mapper.map(ai_text, documents)
        conflicts = self.conflict_detector.detect(documents)

        # Aggregate indexes for quick access
        entities_index: Dict[str, List[Dict[str, Any]]] = {}
        for doc in documents:
            analysis = doc.get("analysis") or {}
            ents = analysis.get("entities") or {}
            for kind, values in ents.items():
                for v in values:
                    entities_index.setdefault(kind, []).append({
                        "value": v,
                        "file_index": doc.get("file_index"),
                        "filename": doc.get("filename"),
                    })

        # Quality report
        quality = self.quality_pipeline.evaluate(ai_text, documents, template, provenance)

        artifacts: Dict[str, Any] = {
            "combined_snippets": combined_snippets,
            "total_pages": total_pages,
            "documents": documents,
            "conflicts": conflicts,
            "provenance": provenance,
            "entities_index": entities_index,
            "template": template,
            "quality": quality,
        }
        return ai_text, artifacts 