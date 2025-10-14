# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for domain service.

This module tests the DomainService class and its methods.
"""

import pytest

from extraction.domain_service import DomainService


class TestDomainService:
    """Test DomainService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.domain_service = DomainService()
    
    def test_normalize_domain(self):
        """Test domain normalization."""
        # Valid domains
        assert self.domain_service.normalize_domain("invoice") == "invoice"
        assert self.domain_service.normalize_domain("INVOICE") == "invoice"
        assert self.domain_service.normalize_domain("  invoice  ") == "invoice"
        
        # Invalid domains default to general
        assert self.domain_service.normalize_domain("invalid_domain") == "invalid_domain"
        assert self.domain_service.normalize_domain("") == "general"
        assert self.domain_service.normalize_domain(None) == "general"
        assert self.domain_service.normalize_domain("   ") == "general"
    
    def test_validate_selected_fields(self):
        """Test field validation."""
        # Valid fields for invoice domain
        valid_fields = ["invoice_number", "total_amount", "vendor_name"]
        result = self.domain_service.validate_selected_fields("invoice", valid_fields)
        assert result == valid_fields
        
        # Mixed valid/invalid fields
        mixed_fields = ["invoice_number", "invalid_field", "total_amount"]
        result = self.domain_service.validate_selected_fields("invoice", mixed_fields)
        assert result == ["invoice_number", "total_amount"]
        
        # Empty fields list returns all allowed fields
        result = self.domain_service.validate_selected_fields("invoice", [])
        assert isinstance(result, list)
        assert "invoice_number" in result
        assert "total_amount" in result
        
        # None fields list returns all allowed fields
        result = self.domain_service.validate_selected_fields("invoice", None)
        assert isinstance(result, list)
        assert "invoice_number" in result
        
        # Invalid domain returns empty list
        result = self.domain_service.validate_selected_fields("invalid_domain", ["field1"])
        assert result == []
    
    def test_get_domain_schema(self):
        """Test getting domain schema."""
        # Valid domain
        schema = self.domain_service.get_domain_schema("invoice")
        assert isinstance(schema, list)
        assert "invoice_number" in schema
        assert "total_amount" in schema
        assert "vendor_name" in schema
        
        # Invalid domain returns general schema
        schema = self.domain_service.get_domain_schema("invalid_domain")
        assert isinstance(schema, list)
        assert "summary" in schema
        assert "emails" in schema
    
    def test_get_all_domains(self):
        """Test getting all domains."""
        domains = self.domain_service.get_all_domains()
        assert isinstance(domains, list)
        assert "invoice" in domains
        assert "contract" in domains
        assert "research" in domains
        assert "healthcare" in domains
        assert "general" in domains
        assert len(domains) > 10  # Should have many domains
    
    def test_is_valid_domain(self):
        """Test domain validation."""
        # Valid domains
        assert self.domain_service.is_valid_domain("invoice") is True
        assert self.domain_service.is_valid_domain("contract") is True
        assert self.domain_service.is_valid_domain("research") is True
        
        # Invalid domains
        assert self.domain_service.is_valid_domain("invalid_domain") is False
        assert self.domain_service.is_valid_domain("") is False
        assert self.domain_service.is_valid_domain("nonexistent") is False
    
    def test_get_field_count(self):
        """Test getting field count for domains."""
        # Invoice should have many fields
        count = self.domain_service.get_field_count("invoice")
        assert count > 10
        
        # General should have few fields
        count = self.domain_service.get_field_count("general")
        assert count == 5  # summary, emails, phones, amounts, dates
        
        # Invalid domain should return 0
        count = self.domain_service.get_field_count("invalid_domain")
        assert count == 0
    
    def test_get_common_fields(self):
        """Test getting common fields across domains."""
        # Test with overlapping domains
        domains = ["invoice", "purchase_order"]
        common = self.domain_service.get_common_fields(domains)
        assert isinstance(common, set)
        # Should have some common fields like currency, vendor_name, etc.
        assert len(common) > 0
        
        # Test with non-overlapping domains
        domains = ["invoice", "research"]
        common = self.domain_service.get_common_fields(domains)
        assert isinstance(common, set)
        # Should have few or no common fields
        
        # Test with empty list
        common = self.domain_service.get_common_fields([])
        assert common == set()
        
        # Test with single domain
        common = self.domain_service.get_common_fields(["invoice"])
        assert isinstance(common, set)
        assert len(common) > 0
    
    def test_domain_fields_structure(self):
        """Test that DOMAIN_FIELDS has expected structure."""
        # Check that all domains have field lists
        for domain, fields in self.domain_service.DOMAIN_FIELDS.items():
            assert isinstance(fields, list)
            assert len(fields) > 0
            # All fields should be strings
            for field in fields:
                assert isinstance(field, str)
                assert len(field) > 0
    
    def test_specific_domain_fields(self):
        """Test specific domain field contents."""
        # Test invoice fields
        invoice_fields = self.domain_service.get_domain_schema("invoice")
        expected_invoice_fields = [
            "invoice_number", "total_amount", "vendor_name", "customer_name",
            "subtotal_amount", "tax_amount", "due_date"
        ]
        for field in expected_invoice_fields:
            assert field in invoice_fields
        
        # Test research fields
        research_fields = self.domain_service.get_domain_schema("research")
        expected_research_fields = [
            "title", "authors", "abstract", "methodology", "results",
            "conclusions", "doi", "citations", "references"
        ]
        for field in expected_research_fields:
            assert field in research_fields
        
        # Test healthcare fields
        healthcare_fields = self.domain_service.get_domain_schema("healthcare")
        expected_healthcare_fields = [
            "patient_id", "mrn", "chief_complaint", "history", "assessment",
            "plan", "medications", "allergies", "vitals", "labs"
        ]
        for field in expected_healthcare_fields:
            assert field in healthcare_fields
    
    def test_field_validation_edge_cases(self):
        """Test edge cases for field validation."""
        # Test with None domain
        result = self.domain_service.validate_selected_fields(None, ["field1"])
        assert result == []
        
        # Test with empty string domain
        result = self.domain_service.validate_selected_fields("", ["field1"])
        assert result == []
        
        # Test with non-list selected_fields
        result = self.domain_service.validate_selected_fields("invoice", "not_a_list")
        assert result == []
        
        # Test with mixed types in selected_fields
        result = self.domain_service.validate_selected_fields("invoice", [1, "invoice_number", None])
        assert result == ["invoice_number"]
