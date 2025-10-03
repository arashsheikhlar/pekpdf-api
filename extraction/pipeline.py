# SPDX-License-Identifier: AGPL-3.0-only

from typing import Dict, Any, List, Tuple
import re

class DocumentTypeDetector:
    """Very lightweight detector to guess document type by keyword heuristics.
    Returns one of: invoice, contract, financial, research, healthcare, general
    """

    KEYWORDS = {
        "invoice": ["invoice", "subtotal", "tax", "total due", "bill to", "invoice number", "due date"],
        "contract": ["agreement", "party", "parties", "term", "termination", "governing law", "liability"],
        "financial": ["income statement", "balance sheet", "cash flow", "revenue", "ebitda", "margin"],
        "research": ["abstract", "methodology", "results", "discussion", "references", "dataset", "hypothesis"],
        "healthcare": ["patient", "diagnosis", "treatment", "dosage", "clinical", "contraindication"],
    }

    def detect(self, text: str) -> str:
        if not text:
            return "general"
        lower = text.lower()
        best = (0, "general")
        for dtype, kws in self.KEYWORDS.items():
            score = sum(1 for kw in kws if kw in lower)
            if score > best[0]:
                best = (score, dtype)
        return best[1]


class BasicExtractor:
    """Extract common entities with regex for MVP.
    Extracts: emails, phones, currency_amounts, dates.
    """

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
    # Accept numbers with optional currency symbol or code and optional percent
    AMOUNT_RE = re.compile(r"(?:(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)\s*)?(?:[$€£])?\s?\d{1,3}(?:[\s,]\d{3})*(?:\.\d+)?%?")
    DATE_RE = re.compile(
        r"(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b)",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"emails": [], "phones": [], "amounts": [], "dates": []}
        emails = list({m.group(0) for m in self.EMAIL_RE.finditer(text)})
        phones = list({m.group(0) for m in self.PHONE_RE.finditer(text)})
        amounts = list({m.group(0) for m in self.AMOUNT_RE.finditer(text)})
        dates = list({m.group(0) for m in self.DATE_RE.finditer(text)})
        # Lightweight post-processing
        def safe_sort(arr: List[str]) -> List[str]:
            try:
                return sorted(arr)
            except Exception:
                return arr
        return {
            "emails": safe_sort(emails)[:500],
            "phones": safe_sort(phones)[:500],
            "amounts": safe_sort(amounts)[:500],
            "dates": safe_sort(dates)[:500],
        }


class DomainFieldMapper:
    """Map generic entities/text to domain-specific fields (MVP heuristics)."""

    INVOICE_NUMBER_RE = re.compile(r"\b(?:invoice\s*(?:no\.|number)\s*[:#]?\s*)([A-Za-z0-9\-_/]+)\b", re.IGNORECASE)
    PO_NUMBER_RE = re.compile(r"\b(?:po\s*(?:no\.|number)\s*[:#]?\s*)([A-Za-z0-9\-_/]+)\b", re.IGNORECASE)
    DUE_DATE_RE = re.compile(r"\b(?:due\s*date)\s*[:#]?\s*(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
    TOTAL_RE = re.compile(r"\b(?:total\s*(?:due|amount)?)\s*[:#]?\s*(?:(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)\s*)?(?:[$€£])?\s?\d{1,3}(?:[\s,]\d{3})*(?:\.\d+)?%?\b", re.IGNORECASE)
    SUBTOTAL_RE = re.compile(r"\bsubtotal\b\s*[:#]?\s*(?:(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)\s*)?(?:[$€£])?\s?\d{1,3}(?:[\s,]\d{3})*(?:\.\d+)?%?\b", re.IGNORECASE)
    TAX_RE = re.compile(r"\b(?:tax|vat)\b\s*[:#]?\s*(?:(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR)\s*)?(?:[$€£])?\s?\d{1,3}(?:[\s,]\d{3})*(?:\.\d+)?%?\b", re.IGNORECASE)

    CONTRACT_PARTY_RE = re.compile(r"\bbetween\s+([A-Z][A-Za-z0-9&.,\-\s]+?)\s+and\s+([A-Z][A-Za-z0-9&.,\-\s]+?)\b")
    GOVERNING_LAW_RE = re.compile(r"\bgoverning\s+law\s+of\s+([A-Z][A-Za-z\s]+)\b", re.IGNORECASE)
    TERM_RE = re.compile(r"\bterm\s*(?:of\s*this\s*agreement)?\s*[:]??\s*(\d+\s*(?:months?|years?))\b", re.IGNORECASE)

    FIN_METRIC_RE = re.compile(r"\b(?:revenue|ebitda|gross\s*margin|operating\s*margin|net\s*income)\b[:\s]*(\$?\d[\d,\.]*%?)", re.IGNORECASE)

    DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    CITATION_RE = re.compile(r"\b[A-Z][A-Za-z\-]+\s+et\s+al\.,\s*\d{4}\b")

    # Research numeric/context patterns
    PVAL_RE = re.compile(r"\bp\s*[<=>]\s*(?:0?\.)?\d+", re.IGNORECASE)
    N_RE = re.compile(r"\bn\s*=\s*\d+", re.IGNORECASE)
    CI_RE = re.compile(r"\b(?:CI|confidence\s*interval)\s*\(?\s*(\d{1,3}(?:\.\d+)?)%\s*\)?\s*[:=]?\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?", re.IGNORECASE)
    EFFECT_RE = re.compile(r"\b(?:r|R|beta|β|OR|HR|RR)\s*=\s*-?\d+(?:\.\d+)?", re.IGNORECASE)
    MEAN_SD_RE = re.compile(r"-?\d+(?:\.\d+)?\s*[±\+\-]\s*\d+(?:\.\d+)?")
    PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%\b")
    FIG_RE = re.compile(r"\b(?:Figure|Fig\.)\s*\d+[A-Za-z]?", re.IGNORECASE)
    TAB_RE = re.compile(r"\bTable\s*\d+[A-Za-z]?", re.IGNORECASE)

    # Healthcare-specific patterns
    # Patient identifiers
    PATIENT_ID_RE = re.compile(r"\b(?:patient\s*(?:id|number|no\.? )\s*[:#]?\s*)([A-Za-z0-9\-_/]+)\b", re.IGNORECASE)
    MRN_RE = re.compile(r"\b(?:MRN|medical\s*record\s*(?:number|no\.?))\s*[:#]?\s*([A-Za-z0-9\-_/]+)\b", re.IGNORECASE)

    # ICD codes (ICD-9, ICD-10)
    ICD9_RE = re.compile(r"\b[VE]?\d{3}\.??\d{0,2}\b")
    ICD10_RE = re.compile(r"\b[A-Z]\d{2}\.?[A-Z0-9]{0,4}\b")

    # CPT codes (prefixed by 'CPT')
    CPT_RE = re.compile(r"\bCPT\s*[:#]?\s*(\d{5})\b", re.IGNORECASE)

    # Medications and dosages/frequencies
    DOSAGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|units?)\b", re.IGNORECASE)
    FREQUENCY_RE = re.compile(r"\b(?:once|twice|three\s*times?|four\s*times?|\d+\s*times?)\s*(?:daily|per\s*day|a\s*day|BID|TID|QID|QD|PRN)\b", re.IGNORECASE)
    MEDICATION_LINE_RE = re.compile(r"^[\-\*•]?\s*([A-Z][A-Za-z0-9\- ]{2,40})\s*(?:\:|\-|–)?\s*(.*)$")

    # Vital signs
    BP_RE = re.compile(r"\b(?:BP|blood\s*pressure)\s*[:#]?\s*(\d{2,3})\/(\d{2,3})\s*(?:mmHg)?\b", re.IGNORECASE)
    TEMP_RE = re.compile(r"\b(?:temp|temperature)\s*[:#]?\s*(\d{2,3}(?:\.\d+)?)\s*[°]?[FC]?\b", re.IGNORECASE)
    HR_RE = re.compile(r"\b(?:HR|heart\s*rate|pulse)\s*[:#]?\s*(\d{2,3})\s*(?:bpm)?\b", re.IGNORECASE)
    RR_RE = re.compile(r"\b(?:RR|respiratory\s*rate)\s*[:#]?\s*(\d{1,2})\s*(?:/min)?\b", re.IGNORECASE)
    O2_RE = re.compile(r"\b(?:O2\s*sat|oxygen\s*saturation|SpO2)\s*[:#]?\s*(\d{2,3})\s*%?\b", re.IGNORECASE)

    # Lab results (very lightweight)
    LAB_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_/%\-]{1,12})\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L|mEq/L|g/dL|%|k/μL|U/L|IU/L)?\b", re.IGNORECASE)

    # Healthcare section headings
    CHIEF_COMPLAINT_HEADINGS = ["chief complaint", "presenting complaint", "cc"]
    HISTORY_HEADINGS = ["history of present illness", "hpi", "history", "medical history"]
    PHYSICAL_EXAM_HEADINGS = ["physical examination", "physical exam", "pe", "examination"]
    ASSESSMENT_HEADINGS = ["assessment", "impression", "diagnosis"]
    PLAN_HEADINGS = ["plan", "treatment plan", "management"]
    MEDICATIONS_HEADINGS = ["medications", "current medications", "meds"]
    ALLERGIES_HEADINGS = ["allergies", "drug allergies"]
    VITALS_HEADINGS = ["vital signs", "vitals"]
    LABS_HEADINGS = ["laboratory results", "lab results", "labs"]

    # Research section heading variants
    ABSTRACT_HEADINGS = ["abstract"]
    METHODS_HEADINGS = ["methods", "method", "methodology", "materials and methods", "experimental", "study design"]
    RESULTS_HEADINGS = ["results", "findings"]
    CONCLUSION_HEADINGS = ["conclusion", "conclusions", "discussion and conclusion", "summary"]
    REFERENCES_HEADINGS = ["references", "bibliography"]
    AFFILIATIONS_HEADINGS = ["affiliations", "author information", "author affiliations"]
    COMMON_HEADINGS = (
        ABSTRACT_HEADINGS
        + METHODS_HEADINGS
        + RESULTS_HEADINGS
        + CONCLUSION_HEADINGS
        + REFERENCES_HEADINGS
        + CHIEF_COMPLAINT_HEADINGS
        + HISTORY_HEADINGS
        + PHYSICAL_EXAM_HEADINGS
        + ASSESSMENT_HEADINGS
        + PLAN_HEADINGS
        + MEDICATIONS_HEADINGS
        + ALLERGIES_HEADINGS
        + VITALS_HEADINGS
        + LABS_HEADINGS
        + ["introduction", "discussion", "acknowledgements", "acknowledgments", "keywords"]
    )

    NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,2})\b")
    AFFIL_RE = re.compile(r"\b(University|Institute|Department|Laboratory|Lab\.|School|College|Hospital|Center|Centre|Faculty|Clinic|Inc\.|Ltd\.|LLC|GmbH|S\.A\.)\b", re.IGNORECASE)

    def _find_heading_positions(self, text: str, headings: List[str]) -> List[int]:
        positions: List[int] = []
        if not text:
            return positions
        for h in headings:
            # Match heading on its own line, optional numbering and colon
            pattern = re.compile(rf"(?mi)^\s*(?:\d+\.?\s*)?{re.escape(h)}\s*:?\s*$", re.IGNORECASE)
            for m in pattern.finditer(text):
                positions.append(m.start())
        positions.sort()
        return positions

    def _extract_section(self, text: str, start_headings: List[str]) -> str:
        if not text:
            return ""
        starts = self._find_heading_positions(text, start_headings)
        if not starts:
            return ""
        start = starts[0]
        # find next section heading after start among common headings
        any_next_positions = [pos for pos in self._find_heading_positions(text, self.COMMON_HEADINGS) if pos > start]
        end = min(any_next_positions) if any_next_positions else len(text)
        section = text[start:end]
        # strip the heading line itself
        section = re.sub(r"(?mi)^\s*.*\n", "", section, count=1)
        # collapse excessive whitespace
        section = re.sub(r"\n{3,}", "\n\n", section).strip()
        return section[:10000]

    def _extract_authors_affiliations(self, text: str) -> Tuple[List[str], List[str]]:
        if not text:
            return [], []
        # Prefer explicit affiliations section if present
        affil_section = self._extract_section(text, self.AFFILIATIONS_HEADINGS)
        affiliations: List[str] = []
        if affil_section:
            # split by lines containing organization keywords
            for ln in affil_section.split('\n'):
                if self.AFFIL_RE.search(ln):
                    s = re.sub(r"\s+", " ", ln).strip()
                    if s:
                        affiliations.append(s)
        else:
            # fallback: scan document lines for affiliation-like lines
            for ln in text.split('\n')[:400]:  # restrict to top part for speed
                if self.AFFIL_RE.search(ln):
                    s = re.sub(r"\s+", " ", ln).strip()
                    if s:
                        affiliations.append(s)
        # Deduplicate, preserve order
        seen = set()
        affiliations_unique: List[str] = []
        for a in affiliations:
            if a not in seen:
                seen.add(a)
                affiliations_unique.append(a)

        # Authors: take early text up to Abstract heading
        abstract_pos = None
        starts = self._find_heading_positions(text, self.ABSTRACT_HEADINGS)
        if starts:
            abstract_pos = starts[0]
        head_text = text[:abstract_pos] if abstract_pos else text[:2000]
        candidates = []
        for m in self.NAME_RE.finditer(head_text):
            name = m.group(1)
            # filter noisy matches
            if len(name.split()) <= 4 and not self.AFFIL_RE.search(name) and name.lower() not in ("abstract", "introduction"):
                candidates.append(name)
        # Deduplicate while preserving order
        seen = set()
        authors: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                authors.append(c)
        return authors[:30], affiliations_unique[:30]

    def map_fields(self, dtype: str, text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        dtype = (dtype or 'general').lower()
        fields: Dict[str, Any] = {}

        if dtype == 'invoice':
            inv = self.INVOICE_NUMBER_RE.search(text or '')
            po = self.PO_NUMBER_RE.search(text or '')
            due = self.DUE_DATE_RE.search(text or '')
            total = self.TOTAL_RE.search(text or '')
            subtotal = self.SUBTOTAL_RE.search(text or '')
            tax = self.TAX_RE.search(text or '')
            fields = {
                'invoice_number': inv.group(1) if inv else None,
                'po_number': po.group(1) if po else None,
                'due_date': due.group(1) if due else None,
                'total_amount': total.group(0) if total else None,
                'subtotal_amount': subtotal.group(0) if subtotal else None,
                'tax_amount': tax.group(0) if tax else None,
                'vendor_email': (entities.get('emails') or [None])[0],
            }
        elif dtype == 'contract':
            parties = self.CONTRACT_PARTY_RE.search(text or '')
            law = self.GOVERNING_LAW_RE.search(text or '')
            term = self.TERM_RE.search(text or '')
            fields = {
                'party_a': parties.group(1).strip() if parties else None,
                'party_b': parties.group(2).strip() if parties else None,
                'governing_law': law.group(1).strip() if law else None,
                'term': term.group(1) if term else None,
                'contact_email': (entities.get('emails') or [None])[0],
            }
        elif dtype == 'financial':
            metrics = []
            # Gate heavy scan behind quick keyword check
            if re.search(r"revenue|ebitda|margin|net\s*income", (text or ''), flags=re.IGNORECASE):
                for m in self.FIN_METRIC_RE.finditer(text or ''):
                    metrics.append(m.group(0))
            fields = {
                'key_metrics': metrics[:100],
            }
        elif dtype == 'research':
            doi = self.DOI_RE.search(text or '')
            citations = []
            if len(text or '') > 4000 or re.search(r"et\s+al\.|\(\d{4}\)", (text or ''), flags=re.IGNORECASE):
                citations = list({m.group(0) for m in self.CITATION_RE.finditer(text or '')})
            # Sections
            abstract = self._extract_section(text, self.ABSTRACT_HEADINGS)
            methods = self._extract_section(text, self.METHODS_HEADINGS)
            results = self._extract_section(text, self.RESULTS_HEADINGS)
            conclusion = self._extract_section(text, self.CONCLUSION_HEADINGS)
            # References: parse lines from references section
            refs_section = self._extract_section(text, self.REFERENCES_HEADINGS)
            references: List[str] = []
            if refs_section:
                for ln in refs_section.split('\n'):
                    s = re.sub(r"\s+", " ", ln).strip()
                    if not s:
                        continue
                    # simple filters for reference-like lines
                    if re.match(r"^\[?\d+\]?\.?\s+", s) or re.search(r"\(\d{4}\)\.|\d{4};", s) or re.search(r"\bdoi\b|10\.\d{4,9}/", s, flags=re.IGNORECASE):
                        references.append(s)
            # Authors and affiliations
            authors, affiliations = self._extract_authors_affiliations(text)
            # Collect domain-relevant metrics
            pvals = list({m.group(0) for m in self.PVAL_RE.finditer(text or '')})
            ns = list({m.group(0) for m in self.N_RE.finditer(text or '')})
            cis = list({m.group(0) for m in self.CI_RE.finditer(text or '')})
            effects = list({m.group(0) for m in self.EFFECT_RE.finditer(text or '')})
            mean_sds = list({m.group(0) for m in self.MEAN_SD_RE.finditer(text or '')})
            pcts = list({m.group(0) for m in self.PCT_RE.finditer(text or '')})
            figs = list({m.group(0) for m in self.FIG_RE.finditer(text or '')})
            tabs = list({m.group(0) for m in self.TAB_RE.finditer(text or '')})
            fields = {
                'doi': doi.group(0) if doi else None,
                'citations': sorted(citations)[:200],
                'authors': authors or None,
                'affiliations': affiliations or None,
                'abstract': abstract or None,
                'methodology': methods or None,
                'results': results or None,
                'conclusions': conclusion or None,
                'references': references[:200] or None,
                'research_metrics': {
                    'p_values': sorted(pvals)[:200],
                    'sample_sizes': sorted(ns)[:200],
                    'confidence_intervals': sorted(cis)[:200],
                    'effect_sizes': sorted(effects)[:200],
                    'means_sd': sorted(mean_sds)[:200],
                    'percentages': sorted(pcts)[:200],
                    'figures': sorted(figs)[:200],
                    'tables': sorted(tabs)[:200],
                }
            }
        elif dtype == 'healthcare':
            # Extract healthcare-specific information
            # IDs
            pid = self.PATIENT_ID_RE.search(text or '')
            mrn = self.MRN_RE.search(text or '')

            # Codes
            icd9: List[str] = []
            icd10: List[str] = []
            cpt: List[str] = []
            if re.search(r"ICD|diagnosis|DX", (text or ''), flags=re.IGNORECASE):
                icd9 = list({m.group(0) for m in self.ICD9_RE.finditer(text or '')})
                icd10 = list({m.group(0) for m in self.ICD10_RE.finditer(text or '')})
            if re.search(r"CPT|procedure|therapy|treatment", (text or ''), flags=re.IGNORECASE):
                cpt = list({m.group(1) for m in self.CPT_RE.finditer(text or '')})

            # Sections
            chief_complaint = self._extract_section(text, self.CHIEF_COMPLAINT_HEADINGS)
            history = self._extract_section(text, self.HISTORY_HEADINGS)
            physical_exam = self._extract_section(text, self.PHYSICAL_EXAM_HEADINGS)
            assessment = self._extract_section(text, self.ASSESSMENT_HEADINGS)
            plan = self._extract_section(text, self.PLAN_HEADINGS)
            meds_section = self._extract_section(text, self.MEDICATIONS_HEADINGS)
            allergies = self._extract_section(text, self.ALLERGIES_HEADINGS)
            vitals_section = self._extract_section(text, self.VITALS_HEADINGS)
            labs_section = self._extract_section(text, self.LABS_HEADINGS)

            # Parse medications from meds_section lines
            medications: List[Dict[str, Any]] = []
            if meds_section:
                for ln in meds_section.split('\n')[:200]:
                    ln = ln.strip()
                    if not ln:
                        continue
                    mm = self.MEDICATION_LINE_RE.match(ln)
                    if not mm:
                        continue
                    name, rest = mm.group(1).strip(), (mm.group(2) or '').strip()
                    dose = None
                    freq = None
                    d = self.DOSAGE_RE.search(rest)
                    if d:
                        dose = d.group(0)
                    f = self.FREQUENCY_RE.search(rest)
                    if f:
                        freq = f.group(0)
                    medications.append({"name": name, "dosage": dose, "frequency": freq, "raw": ln})
            medications = medications[:100]

            # Parse vitals
            vitals: Dict[str, Any] = {}
            if vitals_section:
                bp = self.BP_RE.search(vitals_section)
                if bp:
                    vitals['blood_pressure'] = f"{bp.group(1)}/{bp.group(2)} mmHg"
                t = self.TEMP_RE.search(vitals_section)
                if t:
                    vitals['temperature'] = t.group(1)
                hr = self.HR_RE.search(vitals_section)
                if hr:
                    vitals['heart_rate'] = hr.group(1)
                rr = self.RR_RE.search(vitals_section)
                if rr:
                    vitals['respiratory_rate'] = rr.group(1)
                o2 = self.O2_RE.search(vitals_section)
                if o2:
                    vitals['oxygen_saturation'] = f"{o2.group(1)}%"

            # Parse labs
            labs: List[Dict[str, Any]] = []
            if labs_section:
                for m in self.LAB_VALUE_RE.finditer(labs_section):
                    labs.append({
                        "name": m.group(1),
                        "value": m.group(2),
                        "unit": m.group(3) or None,
                    })
            labs = labs[:200]

            # Free-text diagnoses/procedures (best-effort)
            diag_text = None
            if assessment:
                # first sentence or line
                diag_text = assessment.split('\n')[0][:400]

            procedures_text = None
            if plan:
                for cand in plan.split('\n'):
                    if re.search(r"procedure|therapy|treatment", cand, flags=re.IGNORECASE):
                        procedures_text = cand.strip()[:400]
                        break

            fields = {
                'patient_id': pid.group(1) if pid else None,
                'mrn': mrn.group(1) if mrn else None,
                'icd9_codes': sorted(icd9)[:100] or None,
                'icd10_codes': sorted(icd10)[:100] or None,
                'cpt_codes': sorted(cpt)[:100] or None,
                'chief_complaint': chief_complaint or None,
                'history': history or None,
                'physical_exam': physical_exam or None,
                'assessment': assessment or None,
                'plan': plan or None,
                'medications': medications or None,
                'allergies': allergies or None,
                'vitals': vitals or None,
                'labs': labs or None,
                'diagnosis_text': diag_text or None,
                'procedures_text': procedures_text or None,
                'primary_contact': (entities.get('emails') or [None])[0],
            }
        else:
            fields = {}

        # Remove empty fields
        return {k: v for k, v in fields.items() if v} 


class Validator:
    """Validate mapped fields per domain; return errors and warnings lists."""

    def _safe_float(self, s: str):
        try:
            return float(s)
        except Exception:
            return None

    def _parse_p_value(self, s: str):
        # examples: p<0.05, p = 0.001, p<=.1
        m = re.search(r"p\s*([<>=]=?)\s*(\d*\.?\d+)", s, flags=re.IGNORECASE)
        if not m:
            return None, None
        op, val = m.group(1), m.group(2)
        f = self._safe_float(val)
        return op, f

    def _parse_percent(self, s: str):
        m = re.search(r"(\d*\.?\d+)\s?%", s)
        if not m:
            return None
        return self._safe_float(m.group(1))

    def _parse_ci(self, s: str):
        # CI 95% [low, high]
        m = re.search(r"(?:CI|confidence\s*interval).*?(\d+\.?\d*)%.*?(-?\d+\.?\d*).*?(-?\d+\.?\d*)", s, flags=re.IGNORECASE)
        if not m:
            return None
        level = self._safe_float(m.group(1))
        low = self._safe_float(m.group(2))
        high = self._safe_float(m.group(3))
        return level, low, high

    def validate(self, dtype: str, mapped: Dict[str, Any]) -> Dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        dtype = (dtype or 'general').lower()

        if dtype == 'research':
            # DOI exists and looks valid
            doi = mapped.get('doi')
            if doi and not re.match(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", doi, flags=re.IGNORECASE):
                warnings.append(f"DOI format suspicious: {doi}")

            metrics = mapped.get('research_metrics') or {}
            for pv in metrics.get('p_values', []) or []:
                op, f = self._parse_p_value(pv)
                if f is None:
                    warnings.append(f"Unparsable p-value: {pv}")
                else:
                    if f < 0 or f > 1:
                        errors.append(f"Invalid p-value (not in [0,1]): {pv}")

            for pct in metrics.get('percentages', []) or []:
                f = self._parse_percent(pct)
                if f is None:
                    warnings.append(f"Unparsable percentage: {pct}")
                else:
                    if f < 0 or f > 100:
                        errors.append(f"Percentage out of bounds [0,100]: {pct}")

            for ci in metrics.get('confidence_intervals', []) or []:
                parsed = self._parse_ci(ci)
                if not parsed:
                    warnings.append(f"Unparsable CI: {ci}")
                else:
                    level, low, high = parsed
                    if low is not None and high is not None and low > high:
                        errors.append(f"CI low>high: {ci}")

            for n in metrics.get('sample_sizes', []) or []:
                m = re.search(r"n\s*=\s*(\d+)", n, flags=re.IGNORECASE)
                if not m:
                    warnings.append(f"Unparsable sample size: {n}")
                else:
                    val = int(m.group(1))
                    if val <= 0:
                        errors.append(f"Non-positive sample size: {n}")

            # Section presence and basic sanity
            abstract = mapped.get('abstract')
            if not abstract:
                warnings.append("Abstract not detected")
            else:
                alen = len(abstract)
                if alen < 100:
                    warnings.append("Abstract very short (<100 chars)")
                if alen > 8000:
                    warnings.append("Abstract very long (>8000 chars)")

            if not mapped.get('authors'):
                warnings.append("Authors not detected")
            if not mapped.get('methodology'):
                warnings.append("Methodology section not detected")
            if not mapped.get('results'):
                warnings.append("Results section not detected")
            if not mapped.get('conclusions'):
                warnings.append("Conclusions section not detected")
            refs = mapped.get('references') or []
            if isinstance(refs, list) and len(refs) < 3:
                warnings.append("Few references detected (<3)")

        if dtype == 'invoice':
            total = mapped.get('total_amount')
            if total:
                # strip currency and parse
                m = re.search(r"([\d,]+\.?\d*)", total)
                if m:
                    try:
                        float(m.group(1).replace(',', ''))
                    except Exception:
                        warnings.append(f"Unparsable total_amount: {total}")
            due = mapped.get('due_date')
            if due and not re.search(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}$", due):
                warnings.append(f"Due date format not recognized: {due}")

            # Cross-field: subtotal + tax ≈ total
            st = mapped.get('subtotal_amount')
            tx = mapped.get('tax_amount')
            if total and (st or tx):
                def to_num(s: str):
                    m = re.search(r"([\d,]+\.?\d*)", s or '')
                    if not m:
                        return None
                    try:
                        return float(m.group(1).replace(',', ''))
                    except Exception:
                        return None
                tot_v = to_num(total)
                st_v = to_num(st) if st else 0.0
                tx_v = to_num(tx) if tx else 0.0
                if tot_v is not None:
                    if st is None and tx is None:
                        pass
                    else:
                        if abs((st_v + tx_v) - tot_v) > 0.02:
                            warnings.append("Subtotal + tax does not match total amount")
                        elif st is not None and tx is not None:
                            # Consistent
                            pass

        if dtype == 'healthcare':
            # Validate vitals ranges
            vit = mapped.get('vitals') or {}
            try:
                bp = vit.get('blood_pressure')
                if bp and re.search(r"(\d{2,3})/(\d{2,3})", bp):
                    s = int(re.search(r"(\d{2,3})/(\d{2,3})", bp).group(1))
                    d = int(re.search(r"(\d{2,3})/(\d{2,3})", bp).group(2))
                    if not (70 <= s <= 220 and 40 <= d <= 140):
                        warnings.append(f"Blood pressure out of expected range: {bp}")
                t = vit.get('temperature')
                if t:
                    tf = float(str(t))
                    if not (30 <= tf <= 43):
                        warnings.append(f"Temperature out of expected range: {t}")
                hr = vit.get('heart_rate')
                if hr:
                    hv = int(str(hr))
                    if not (30 <= hv <= 220):
                        warnings.append(f"Heart rate out of expected range: {hr}")
                rr = vit.get('respiratory_rate')
                if rr:
                    rv = int(str(rr))
                    if not (6 <= rv <= 40):
                        warnings.append(f"Respiratory rate out of expected range: {rr}")
                o2 = vit.get('oxygen_saturation')
                if o2 and re.search(r"(\d{2,3})", o2):
                    ov = int(re.search(r"(\d{2,3})", o2).group(1))
                    if not (50 <= ov <= 100):
                        warnings.append(f"Oxygen saturation out of expected range: {o2}")
            except Exception:
                pass

            # Basic checks for codes length
            for code_field in ('icd9_codes', 'icd10_codes', 'cpt_codes'):
                arr = mapped.get(code_field) or []
                if isinstance(arr, list) and len(arr) > 200:
                    warnings.append(f"Too many codes in {code_field}: {len(arr)}")

        return {"errors": errors, "warnings": warnings} 


class ConfidenceScorer:
    """Compute simple confidence scores (0-100) for mapped fields based on patterns and validation."""

    def _score_amount(self, s: str) -> int:
        if not s:
            return 0
        # If parseable numeric amount
        m = re.search(r"([\d,]+\.?\d*)", s)
        if not m:
            return 40
        try:
            float(m.group(1).replace(',', ''))
            return 75
        except Exception:
            return 50

    def _has_currency(self, s: str) -> bool:
        return bool(re.search(r"USD|EUR|GBP|CAD|AUD|CHF|JPY|INR|[$€£]", s or '', flags=re.IGNORECASE))

    def _is_iso_date(self, s: str) -> bool:
        return bool(re.search(r"^\d{4}-\d{2}-\d{2}$", s or ''))

    def score(self, dtype: str, mapped: Dict[str, Any], entities: Dict[str, Any], validation: Dict[str, Any], provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
        dtype = (dtype or 'general').lower()
        fields: Dict[str, int] = {}

        warns = set((validation or {}).get('warnings') or [])
        errors = set((validation or {}).get('errors') or [])

        if dtype == 'invoice':
            fields['invoice_number'] = 85 if mapped.get('invoice_number') else 0
            inv = (mapped.get('invoice_number') or '')
            if fields['invoice_number'] and re.match(r"^(INV|Invoice)[-_]?\w+", inv, flags=re.IGNORECASE):
                fields['invoice_number'] = min(95, fields['invoice_number'] + 5)
            fields['po_number'] = 70 if mapped.get('po_number') else 0
            fields['due_date'] = 70 if mapped.get('due_date') else 0
            if self._is_iso_date(mapped.get('due_date') or ''):
                fields['due_date'] = min(90, max(fields['due_date'], 80))
            fields['total_amount'] = self._score_amount(mapped.get('total_amount') or '')
            if self._has_currency(mapped.get('total_amount') or ''):
                fields['total_amount'] = min(90, fields['total_amount'] + 10)
            if any('total_amount' in w for w in warns):
                fields['total_amount'] = max(30, fields['total_amount'] - 20)
            fields['subtotal_amount'] = self._score_amount(mapped.get('subtotal_amount') or '')
            if self._has_currency(mapped.get('subtotal_amount') or ''):
                fields['subtotal_amount'] = min(90, fields['subtotal_amount'] + 8)
            fields['tax_amount'] = self._score_amount(mapped.get('tax_amount') or '')
            if self._has_currency(mapped.get('tax_amount') or ''):
                fields['tax_amount'] = min(90, fields['tax_amount'] + 5)
            if any('Subtotal + tax does not match' in w for w in warns):
                # penalize if mismatch
                fields['subtotal_amount'] = max(20, fields['subtotal_amount'] - 25)
                fields['tax_amount'] = max(20, fields['tax_amount'] - 25)
        elif dtype == 'research':
            # DOI
            fields['doi'] = 90 if mapped.get('doi') else 0
            if provenance and isinstance(provenance.get('doi_pages'), list) and provenance.get('doi_pages'):
                fields['doi'] = min(95, max(80, fields['doi'] + 5))
            # Sections
            def sec_score(txt: str, short_warn: str) -> int:
                if not txt:
                    return 0
                ln = len(txt)
                base = 70 if ln >= 100 else 40
                if any(short_warn in w for w in warns):
                    base -= 15
                return max(0, min(90, base))
            fields['abstract'] = sec_score(mapped.get('abstract') or '', 'Abstract very short')
            fields['methodology'] = 70 if mapped.get('methodology') else 0
            fields['results'] = 70 if mapped.get('results') else 0
            fields['conclusions'] = 65 if mapped.get('conclusions') else 0
            if provenance and isinstance(provenance.get('sections'), dict):
                for k in ('abstract','methodology','results','conclusions'):
                    pages = provenance['sections'].get(k)
                    if fields.get(k, 0) and isinstance(pages, list) and pages:
                        fields[k] = min(95, fields[k] + 5)
            # Authors/Affiliations
            fields['authors'] = 60 if (mapped.get('authors') or []) else 0
            fields['affiliations'] = 60 if (mapped.get('affiliations') or []) else 0
            # References/citations
            refs = mapped.get('references') or []
            fields['references'] = 75 if isinstance(refs, list) and len(refs) >= 5 else (50 if refs else 0)
            cits = mapped.get('citations') or []
            fields['citations'] = 60 if cits else 0
            if provenance and isinstance(provenance.get('citations'), list):
                cited_hits = sum(1 for c in provenance['citations'] if (c.get('pages') or []))
                if cited_hits >= 5:
                    fields['citations'] = min(85, max(fields['citations'], 70))
            # Metrics
            metrics = mapped.get('research_metrics') or {}
            present_metrics = sum(1 for k in ['p_values','sample_sizes','confidence_intervals','effect_sizes','means_sd','percentages'] if (metrics.get(k) or []))
            fields['research_metrics'] = 60 + 5*present_metrics if present_metrics else 0
            if provenance and isinstance(provenance.get('metrics'), dict):
                prov_hits = 0
                for name, entries in provenance['metrics'].items():
                    if isinstance(entries, list):
                        prov_hits += sum(1 for e in entries if (e.get('pages') or []))
                if prov_hits >= 5:
                    fields['research_metrics'] = min(90, max(fields['research_metrics'], 70))
        elif dtype == 'financial':
            km = mapped.get('key_metrics') or []
            fields['key_metrics'] = 60 if km else 0
        elif dtype == 'contract':
            fields['party_a'] = 70 if mapped.get('party_a') else 0
            fields['party_b'] = 70 if mapped.get('party_b') else 0
            fields['governing_law'] = 70 if mapped.get('governing_law') else 0
            fields['term'] = 60 if mapped.get('term') else 0
        elif dtype == 'healthcare':
            # IDs
            fields['patient_id'] = 75 if mapped.get('patient_id') else 0
            fields['mrn'] = 80 if mapped.get('mrn') else 0
            # Codes
            fields['icd9_codes'] = 60 if (mapped.get('icd9_codes') or []) else 0
            fields['icd10_codes'] = 70 if (mapped.get('icd10_codes') or []) else 0
            fields['cpt_codes'] = 65 if (mapped.get('cpt_codes') or []) else 0
            # Sections
            fields['chief_complaint'] = 65 if mapped.get('chief_complaint') else 0
            fields['history'] = 65 if mapped.get('history') else 0
            fields['physical_exam'] = 65 if mapped.get('physical_exam') else 0
            fields['assessment'] = 70 if mapped.get('assessment') else 0
            fields['plan'] = 65 if mapped.get('plan') else 0
            # Meds
            fields['medications'] = 60 if (mapped.get('medications') or []) else 0
            fields['allergies'] = 55 if mapped.get('allergies') else 0
            # Vitals
            vit = mapped.get('vitals') or {}
            vit_present = any(vit.get(k) for k in ('blood_pressure','temperature','heart_rate','respiratory_rate','oxygen_saturation'))
            fields['vitals'] = 65 if vit_present else 0
            # Labs
            fields['labs'] = 60 if (mapped.get('labs') or []) else 0
            # Free text
            fields['diagnosis_text'] = 60 if mapped.get('diagnosis_text') else 0
            fields['procedures_text'] = 55 if mapped.get('procedures_text') else 0
            # Contact
            fields['primary_contact'] = 50 if mapped.get('primary_contact') else 0
 
        # overall: average of non-zero fields, else 0
        vals = [v for v in fields.values() if v > 0]
        overall = int(sum(vals)/len(vals)) if vals else 0
        # Penalize overall for validation problems
        overall = max(0, overall - min(20, len(errors)*10 + len(warns)*3))
        return { 'overall': overall, 'fields': fields } 