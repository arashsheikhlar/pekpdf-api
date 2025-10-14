# SPDX-License-Identifier: AGPL-3.0-only

"""
Domain service for document type management and field validation.

This module handles document type normalization, field validation, and schema management
for the extraction system.
"""

from typing import Dict, List, Optional, Set
from .models import DocumentType


class DomainService:
    """Service for managing document domains and field schemas."""
    
    # Backend authoritative map of domains to their canonical field keys
    # UI may present categories/groups, but backend validation uses this flat list
    DOMAIN_FIELDS: Dict[str, List[str]] = {
        # Core domains (existing + expanded)
        "invoice": [
            "invoice_number", "po_number", "invoice_date", "due_date", "currency",
            "vendor_name", "vendor_address", "vendor_email", "vendor_phone", "vendor_tax_id",
            "customer_name", "customer_address", "customer_email", "customer_phone", "customer_tax_id",
            "subtotal_amount", "tax_amount", "tax_rate", "shipping_amount", "discount_amount", "total_amount",
            "payment_terms", "payment_method", "notes",
            "line_items"  # list of {description, quantity, unit_price, amount}
        ],
        "purchase_order": [
            "po_number", "requisition_number", "po_date", "delivery_date", "currency",
            "buyer_name", "buyer_address", "buyer_email", "buyer_phone",
            "vendor_name", "vendor_address", "vendor_email", "vendor_phone",
            "ship_to_name", "ship_to_address",
            "bill_to_name", "bill_to_address",
            "payment_terms", "approval_status", "approver_name",
            "subtotal_amount", "tax_amount", "shipping_amount", "discount_amount", "total_amount",
            "line_items"
        ],
        "receipt": [
            "receipt_number", "receipt_date", "store_name", "store_address", "store_phone",
            "cashier_name", "payment_method", "currency", "tax_amount", "total_amount",
            "line_items"
        ],
        "contract": [
            "party_a", "party_b", "effective_date", "term", "termination", "governing_law",
            "jurisdiction", "payment_terms", "confidentiality", "liability", "indemnity",
            "contact_email", "signatories"
        ],
        "financial": [
            "statement_type", "period", "currency",
            "revenue", "cost_of_goods_sold", "gross_profit", "operating_income", "operating_expenses",
            "net_income", "ebitda", "gross_margin", "operating_margin", "net_margin", "eps",
            "operating_cash_flow", "free_cash_flow", "total_assets", "total_liabilities",
            "shareholders_equity", "debt", "cash_and_equivalents", "ratios", "line_items"
        ],
        "bank_statement": [
            "account_holder", "account_number", "statement_period", "opening_balance",
            "closing_balance", "total_deposits", "total_withdrawals", "fees", "interest",
            "transactions"  # list of {date, description, debit, credit, balance}
        ],
        "tax_form": [
            "form_type", "tax_year", "filer_name", "filer_ssn_ein", "filer_address",
            "income_categories", "deductions", "credits", "tax_owed", "refund_amount"
        ],
        "research": [
            "title", "authors", "affiliations", "abstract", "methodology", "results",
            "conclusions", "keywords", "doi", "citations", "references", "research_metrics"
        ],
        "healthcare": [
            "patient_id", "mrn", "patient_name", "dob", "icd9_codes", "icd10_codes", "cpt_codes",
            "chief_complaint", "history", "physical_exam", "assessment", "plan",
            "medications", "allergies", "vitals", "labs", "diagnosis_text", "procedures_text",
            "primary_contact"
        ],
        "resume": [
            "name", "email", "phone", "address", "linkedin", "portfolio",
            "summary", "skills_technical", "skills_soft",
            "experience",  # list of {company, role, start_date, end_date, responsibilities, achievements}
            "education",   # list of {degree, institution, start_date, end_date, gpa}
            "certifications", "languages", "publications", "awards"
        ],
        "legal_pleading": [
            "case_name", "court", "docket_number", "judge", "filing_date",
            "parties", "claims", "relief_requested", "orders"
        ],
        "patent": [
            "patent_number", "application_number", "title", "assignee", "inventors",
            "filing_date", "issue_date", "ipc_codes", "abstract", "claims", "description"
        ],
        "medical_bill": [
            "provider_name", "provider_address", "patient_name", "patient_id", "service_dates",
            "icd_codes", "cpt_codes", "charges", "insurance_payments", "patient_responsibility",
            "total_amount"
        ],
        "lab_report": [
            "patient_name", "patient_id", "collection_date", "report_date", "ordering_physician",
            "tests",  # list of {name, value, unit, reference_range}
            "interpretation"
        ],
        "insurance_claim": [
            "claim_number", "policy_number", "insured_name", "loss_date", "loss_description",
            "adjuster_name", "status", "payments", "total_amount"
        ],
        "real_estate": [
            "property_address", "parcel_number", "owner_name", "assessed_value", "sale_price",
            "mortgage_lender", "loan_amount", "closing_date", "recording_number"
        ],
        "shipping_manifest": [
            "manifest_number", "carrier", "ship_date", "origin", "destination",
            "items",  # list of {description, quantity, weight, value}
            "total_weight", "total_value"
        ],
        # Generic fallback
        "general": [
            "summary", "emails", "phones", "amounts", "dates"
        ],
    }
    
    def __init__(self):
        """Initialize the domain service."""
        pass
    
    def normalize_domain(self, value: Optional[str]) -> str:
        """
        Normalize domain value to a valid document type.
        
        Args:
            value: Raw domain string (can be None)
            
        Returns:
            Normalized domain string, defaults to "general" if invalid/empty
        """
        d = (value or "").strip().lower()
        return d if d in self.DOMAIN_FIELDS else ("general" if d == "" else d)
    
    def validate_selected_fields(self, domain: str, selected_fields: Optional[List[str]]) -> List[str]:
        """
        Validate and filter selected fields against domain schema.
        
        Args:
            domain: Document domain/type
            selected_fields: List of field names to validate
            
        Returns:
            List of valid fields for the domain
        """
        allowed = set(self.DOMAIN_FIELDS.get(domain, []))
        if not selected_fields:
            return list(allowed)
        return [f for f in selected_fields if f in allowed]
    
    def get_domain_schema(self, domain: str) -> List[str]:
        """
        Get the complete field schema for a domain.
        
        Args:
            domain: Document domain/type
            
        Returns:
            List of all valid fields for the domain
        """
        return self.DOMAIN_FIELDS.get(domain, self.DOMAIN_FIELDS["general"])
    
    def get_all_domains(self) -> List[str]:
        """
        Get list of all supported domains.
        
        Returns:
            List of all domain names
        """
        return list(self.DOMAIN_FIELDS.keys())
    
    def is_valid_domain(self, domain: str) -> bool:
        """
        Check if a domain is valid/supported.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is supported
        """
        return domain in self.DOMAIN_FIELDS
    
    def get_field_count(self, domain: str) -> int:
        """
        Get the number of fields for a domain.
        
        Args:
            domain: Document domain/type
            
        Returns:
            Number of fields in the domain schema
        """
        return len(self.get_domain_schema(domain))
    
    def get_common_fields(self, domains: List[str]) -> Set[str]:
        """
        Get fields common to multiple domains.
        
        Args:
            domains: List of domains to compare
            
        Returns:
            Set of fields present in all specified domains
        """
        if not domains:
            return set()
        
        field_sets = [set(self.get_domain_schema(d)) for d in domains]
        return set.intersection(*field_sets) if field_sets else set()
