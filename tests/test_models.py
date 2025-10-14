# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for extraction models.

This module tests the Pydantic models used throughout the extraction system.
"""

import pytest
from pydantic import ValidationError

from extraction.models import (
    DocumentType,
    ExtractionOptions,
    ExtractionResult,
    ValidationResult,
    ConfidenceScore,
    CacheEntry,
    PluginMetadata,
    ExtractionError
)


class TestDocumentType:
    """Test DocumentType enum."""
    
    def test_document_type_values(self):
        """Test that all expected document types are present."""
        expected_types = [
            "invoice", "purchase_order", "receipt", "contract", "financial",
            "bank_statement", "tax_form", "research", "healthcare", "resume",
            "legal_pleading", "patent", "medical_bill", "lab_report",
            "insurance_claim", "real_estate", "shipping_manifest", "general"
        ]
        
        for doc_type in expected_types:
            assert hasattr(DocumentType, doc_type.upper())
            assert DocumentType(doc_type) == doc_type


class TestExtractionOptions:
    """Test ExtractionOptions model."""
    
    def test_default_values(self):
        """Test default values for ExtractionOptions."""
        options = ExtractionOptions()
        assert options.domain_override is None
        assert options.selected_fields == []
        assert options.custom_instructions == ""
        assert options.enrich is False
        assert options.use_ocr is False
    
    def test_custom_values(self):
        """Test custom values for ExtractionOptions."""
        options = ExtractionOptions(
            domain_override="invoice",
            selected_fields=["invoice_number", "total_amount"],
            custom_instructions="Extract dates in ISO format",
            enrich=True,
            use_ocr=True
        )
        assert options.domain_override == "invoice"
        assert options.selected_fields == ["invoice_number", "total_amount"]
        assert options.custom_instructions == "Extract dates in ISO format"
        assert options.enrich is True
        assert options.use_ocr is True
    
    def test_selected_fields_validation(self):
        """Test selected_fields validation."""
        # Valid list of strings
        options = ExtractionOptions(selected_fields=["field1", "field2"])
        assert options.selected_fields == ["field1", "field2"]
        
        # Empty list
        options = ExtractionOptions(selected_fields=[])
        assert options.selected_fields == []
        
        # Non-list input should be converted to empty list
        options = ExtractionOptions(selected_fields="not_a_list")
        assert options.selected_fields == []
        
        # Mixed types should be converted to strings
        options = ExtractionOptions(selected_fields=[1, "field2", None])
        assert options.selected_fields == ["1", "field2", ""]


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_default_values(self):
        """Test default values for ValidationResult."""
        result = ValidationResult()
        assert result.errors == []
        assert result.warnings == []
    
    def test_custom_values(self):
        """Test custom values for ValidationResult."""
        result = ValidationResult(
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == ["Warning 1"]


class TestConfidenceScore:
    """Test ConfidenceScore model."""
    
    def test_default_values(self):
        """Test default values for ConfidenceScore."""
        score = ConfidenceScore()
        assert score.overall == 0
        assert score.fields == {}
    
    def test_custom_values(self):
        """Test custom values for ConfidenceScore."""
        score = ConfidenceScore(
            overall=85,
            fields={"field1": 90, "field2": 75}
        )
        assert score.overall == 85
        assert score.fields == {"field1": 90, "field2": 75}
    
    def test_overall_range_validation(self):
        """Test overall score range validation."""
        # Valid range
        score = ConfidenceScore(overall=50)
        assert score.overall == 50
        
        # Boundary values
        score = ConfidenceScore(overall=0)
        assert score.overall == 0
        
        score = ConfidenceScore(overall=100)
        assert score.overall == 100
        
        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            ConfidenceScore(overall=-1)
        
        with pytest.raises(ValidationError):
            ConfidenceScore(overall=101)


class TestExtractionResult:
    """Test ExtractionResult model."""
    
    def test_minimal_result(self):
        """Test minimal ExtractionResult."""
        result = ExtractionResult(
            type="invoice",
            pages=5,
            entities={},
            mapped_fields={}
        )
        assert result.type == "invoice"
        assert result.pages == 5
        assert result.entities == {}
        assert result.mapped_fields == {}
        assert result.custom_fields is None
        assert isinstance(result.validation, ValidationResult)
        assert isinstance(result.confidence, ConfidenceScore)
        assert result.tables == []
        assert result.formulas == []
        assert result.provenance is None
    
    def test_full_result(self):
        """Test full ExtractionResult."""
        result = ExtractionResult(
            type="research",
            pages=10,
            entities={"emails": ["test@example.com"]},
            mapped_fields={"title": "Test Paper"},
            custom_fields={"keywords": ["AI", "ML"]},
            validation=ValidationResult(errors=["Error 1"]),
            confidence=ConfidenceScore(overall=85),
            tables=[{"page": 1, "data": []}],
            formulas=[{"page": 2, "formula": "x = y + z"}],
            provenance={"sections": {"abstract": [1]}}
        )
        assert result.type == "research"
        assert result.pages == 10
        assert result.entities == {"emails": ["test@example.com"]}
        assert result.mapped_fields == {"title": "Test Paper"}
        assert result.custom_fields == {"keywords": ["AI", "ML"]}
        assert result.validation.errors == ["Error 1"]
        assert result.confidence.overall == 85
        assert len(result.tables) == 1
        assert len(result.formulas) == 1
        assert result.provenance == {"sections": {"abstract": [1]}}
    
    def test_to_dict_and_from_dict(self):
        """Test serialization methods."""
        result = ExtractionResult(
            type="invoice",
            pages=3,
            entities={"phones": ["123-456-7890"]},
            mapped_fields={"total_amount": "$100.00"}
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["type"] == "invoice"
        assert result_dict["pages"] == 3
        
        # Test from_dict
        new_result = ExtractionResult.from_dict(result_dict)
        assert new_result.type == "invoice"
        assert new_result.pages == 3
        assert new_result.entities == {"phones": ["123-456-7890"]}
        assert new_result.mapped_fields == {"total_amount": "$100.00"}


class TestCacheEntry:
    """Test CacheEntry model."""
    
    def test_cache_entry(self):
        """Test CacheEntry creation and expiration."""
        result = ExtractionResult(
            type="general",
            pages=1,
            entities={},
            mapped_fields={}
        )
        
        entry = CacheEntry(
            result=result,
            timestamp=1000.0,
            ttl=3600
        )
        
        assert entry.result == result
        assert entry.timestamp == 1000.0
        assert entry.ttl == 3600
        
        # Test expiration (mock time)
        import time
        original_time = time.time
        time.time = lambda: 2000.0  # 1000 seconds later
        assert entry.is_expired() is False
        
        time.time = lambda: 5000.0  # 4000 seconds later (expired)
        assert entry.is_expired() is True
        
        # Restore original time function
        time.time = original_time


class TestPluginMetadata:
    """Test PluginMetadata model."""
    
    def test_plugin_metadata(self):
        """Test PluginMetadata creation."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.2.3",
            description="A test plugin",
            supported_types=["invoice", "contract"],
            priority=75
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.2.3"
        assert metadata.description == "A test plugin"
        assert metadata.supported_types == ["invoice", "contract"]
        assert metadata.priority == 75
    
    def test_default_values(self):
        """Test default values for PluginMetadata."""
        metadata = PluginMetadata(name="test")
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == ""
        assert metadata.supported_types == []
        assert metadata.priority == 0


class TestExtractionError:
    """Test ExtractionError model."""
    
    def test_extraction_error(self):
        """Test ExtractionError creation."""
        error = ExtractionError(
            error_type="validation_error",
            message="Invalid field value",
            details={"field": "amount", "value": "invalid"}
        )
        
        assert error.error_type == "validation_error"
        assert error.message == "Invalid field value"
        assert error.details == {"field": "amount", "value": "invalid"}
        assert isinstance(error.timestamp, float)
        assert error.timestamp > 0
