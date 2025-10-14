# SPDX-License-Identifier: AGPL-3.0-only

from typing import Dict, Any
import re


class ConfidenceScorer:
    """Compute simple confidence scores (0-100) for mapped fields based on patterns and validation."""

    def _score_amount(self, s: str) -> int:
        if not s:
            return 0
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

    def score(self, dtype: str, mapped: Dict[str, Any], entities: Dict[str, Any], validation: Dict[str, Any], provenance: Dict[str, Any] | None = None, selected_fields: list[str] | None = None) -> Dict[str, Any]:
        dtype = (dtype or 'general').lower()
        fields: Dict[str, int] = {}

        warns = set((validation or {}).get('warnings') or [])
        errors = set((validation or {}).get('errors') or [])

        # If selected_fields is provided, only score those fields
        if selected_fields:
            # Filter to only include fields that are in selected_fields
            def should_score_field(field_name: str) -> bool:
                return field_name in selected_fields
        else:
            # If no selected_fields, score all fields (backward compatibility)
            def should_score_field(field_name: str) -> bool:
                return True

        if dtype == 'invoice':
            if should_score_field('invoice_number'):
                fields['invoice_number'] = 85 if mapped.get('invoice_number') else 0
                inv = (mapped.get('invoice_number') or '')
                if fields['invoice_number'] and re.match(r"^(INV|Invoice)[-_]?\w+", inv, flags=re.IGNORECASE):
                    fields['invoice_number'] = min(95, fields['invoice_number'] + 5)
            if should_score_field('po_number'):
                fields['po_number'] = 70 if mapped.get('po_number') else 0
            if should_score_field('due_date'):
                fields['due_date'] = 70 if mapped.get('due_date') else 0
                if self._is_iso_date(mapped.get('due_date') or ''):
                    fields['due_date'] = min(90, max(fields['due_date'], 80))
            if should_score_field('total_amount'):
                fields['total_amount'] = self._score_amount(mapped.get('total_amount') or '')
                if self._has_currency(mapped.get('total_amount') or ''):
                    fields['total_amount'] = min(90, fields['total_amount'] + 10)
                if any('total_amount' in w for w in warns):
                    fields['total_amount'] = max(30, fields['total_amount'] - 20)
            if should_score_field('subtotal_amount'):
                fields['subtotal_amount'] = self._score_amount(mapped.get('subtotal_amount') or '')
                if self._has_currency(mapped.get('subtotal_amount') or ''):
                    fields['subtotal_amount'] = min(90, fields['subtotal_amount'] + 8)
            if should_score_field('tax_amount'):
                fields['tax_amount'] = self._score_amount(mapped.get('tax_amount') or '')
                if self._has_currency(mapped.get('tax_amount') or ''):
                    fields['tax_amount'] = min(90, fields['tax_amount'] + 5)
                if any('Subtotal + tax does not match' in w for w in warns):
                    fields['subtotal_amount'] = max(20, fields['subtotal_amount'] - 25)
                    fields['tax_amount'] = max(20, fields['tax_amount'] - 25)
        elif dtype == 'research':
            if should_score_field('doi'):
                fields['doi'] = 90 if mapped.get('doi') else 0
                if provenance and isinstance(provenance.get('doi_pages'), list) and provenance.get('doi_pages'):
                    fields['doi'] = min(95, max(80, fields['doi'] + 5))
            def sec_score(txt: str, short_warn: str) -> int:
                if not txt:
                    return 0
                ln = len(txt)
                base = 70 if ln >= 100 else 40
                if any(short_warn in w for w in warns):
                    base -= 15
                return max(0, min(90, base))
            if should_score_field('abstract'):
                fields['abstract'] = sec_score(mapped.get('abstract') or '', 'Abstract very short')
            if should_score_field('methodology'):
                fields['methodology'] = 70 if mapped.get('methodology') else 0
            if should_score_field('results'):
                fields['results'] = 70 if mapped.get('results') else 0
            if should_score_field('conclusions'):
                fields['conclusions'] = 65 if mapped.get('conclusions') else 0
            if provenance and isinstance(provenance.get('sections'), dict):
                for k in ('abstract','methodology','results','conclusions'):
                    if should_score_field(k):
                        pages = provenance['sections'].get(k)
                        if fields.get(k, 0) and isinstance(pages, list) and pages:
                            fields[k] = min(95, fields[k] + 5)
            if should_score_field('authors'):
                fields['authors'] = 60 if (mapped.get('authors') or []) else 0
            if should_score_field('affiliations'):
                fields['affiliations'] = 60 if (mapped.get('affiliations') or []) else 0
            if should_score_field('references'):
                refs = mapped.get('references') or []
                fields['references'] = 75 if isinstance(refs, list) and len(refs) >= 5 else (50 if refs else 0)
            if should_score_field('citations'):
                cits = mapped.get('citations') or []
                fields['citations'] = 60 if cits else 0
                if provenance and isinstance(provenance.get('citations'), list):
                    cited_hits = sum(1 for c in provenance['citations'] if (c.get('pages') or []))
                    if cited_hits >= 5:
                        fields['citations'] = min(85, max(fields['citations'], 70))
            if should_score_field('research_metrics'):
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
            if should_score_field('key_metrics'):
                km = mapped.get('key_metrics') or []
                fields['key_metrics'] = 60 if km else 0
        elif dtype == 'contract':
            if should_score_field('party_a'):
                fields['party_a'] = 70 if mapped.get('party_a') else 0
            if should_score_field('party_b'):
                fields['party_b'] = 70 if mapped.get('party_b') else 0
            if should_score_field('governing_law'):
                fields['governing_law'] = 70 if mapped.get('governing_law') else 0
            if should_score_field('term'):
                fields['term'] = 60 if mapped.get('term') else 0
        elif dtype == 'healthcare':
            if should_score_field('patient_id'):
                fields['patient_id'] = 75 if mapped.get('patient_id') else 0
            if should_score_field('mrn'):
                fields['mrn'] = 80 if mapped.get('mrn') else 0
            if should_score_field('icd9_codes'):
                fields['icd9_codes'] = 60 if (mapped.get('icd9_codes') or []) else 0
            if should_score_field('icd10_codes'):
                fields['icd10_codes'] = 70 if (mapped.get('icd10_codes') or []) else 0
            if should_score_field('cpt_codes'):
                fields['cpt_codes'] = 65 if (mapped.get('cpt_codes') or []) else 0
            if should_score_field('chief_complaint'):
                fields['chief_complaint'] = 65 if mapped.get('chief_complaint') else 0
            if should_score_field('history'):
                fields['history'] = 65 if mapped.get('history') else 0
            if should_score_field('physical_exam'):
                fields['physical_exam'] = 65 if mapped.get('physical_exam') else 0
            if should_score_field('assessment'):
                fields['assessment'] = 70 if mapped.get('assessment') else 0
            if should_score_field('plan'):
                fields['plan'] = 65 if mapped.get('plan') else 0
            if should_score_field('medications'):
                fields['medications'] = 60 if (mapped.get('medications') or []) else 0
            if should_score_field('allergies'):
                fields['allergies'] = 55 if mapped.get('allergies') else 0
            if should_score_field('vitals'):
                vit = mapped.get('vitals') or []
                vit_present = isinstance(vit, dict) and any(vit.get(k) for k in ('blood_pressure','temperature','heart_rate','respiratory_rate','oxygen_saturation'))
                fields['vitals'] = 65 if vit_present else 0
            if should_score_field('labs'):
                fields['labs'] = 60 if (mapped.get('labs') or []) else 0
            if should_score_field('diagnosis_text'):
                fields['diagnosis_text'] = 60 if mapped.get('diagnosis_text') else 0
            if should_score_field('procedures_text'):
                fields['procedures_text'] = 55 if mapped.get('procedures_text') else 0
            if should_score_field('primary_contact'):
                fields['primary_contact'] = 50 if mapped.get('primary_contact') else 0

        vals = [v for v in fields.values() if v > 0]
        overall = int(sum(vals)/len(vals)) if vals else 0
        overall = max(0, overall - min(20, len(errors)*10 + len(warns)*3))
        return { 'overall': overall, 'fields': fields }

