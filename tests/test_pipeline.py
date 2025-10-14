# SPDX-License-Identifier: AGPL-3.0-only

import pytest
from unittest.mock import Mock, patch, MagicMock
from extraction.pipeline import ExtractionPipeline
from extraction.models import (
    ExtractionOptions, ExtractionResult, DocumentType, 
    ExtractedEntities, MappedFields, ValidationResult, 
    Provenance, CustomFields, ConfidenceScore
)


class TestExtractionPipeline:
    """Test suite for ExtractionPipeline orchestration."""

    @pytest.fixture
    def mock_domain_service(self):
        """Provides a mocked DomainService."""
        service = Mock()
        service.normalize_domain.return_value = "invoice"
        service.validate_selected_fields.return_value = ["invoice_number", "total_amount"]
        return service

    @pytest.fixture
    def mock_ai_service(self):
        """Provides a mocked AIExtractionService."""
        service = Mock()
        service.ai_only_extract.return_value = {
            "type": "invoice",
            "entities": {"emails": ["test@example.com"]},
            "mapped_fields": {"invoice_number": "INV-001", "total_amount": "$100.00"},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        service.enrich_extraction_with_llm.return_value = {
            "invoice_number": "INV-001",
            "total_amount": "$100.00",
            "enrichment": {"summary": "Enhanced extraction"}
        }
        return service

    @pytest.fixture
    def mock_text_service(self):
        """Provides a mocked TextExtractionService."""
        service = Mock()
        service.extract_text.return_value = (
            "Invoice INV-001 Total: $100.00",
            ["Invoice INV-001 Total: $100.00"],
            {"invoice": [0]}
        )
        return service

    @pytest.fixture
    def mock_cache_service(self):
        """Provides a mocked CacheService."""
        service = Mock()
        service.get_cached_result.return_value = None
        service.cache_result.return_value = None
        return service

    @pytest.fixture
    def mock_confidence_scorer(self):
        """Provides a mocked ConfidenceScorer."""
        scorer = Mock()
        scorer.score.return_value = ConfidenceScore(
            overall=85,
            fields={"invoice_number": 90, "total_amount": 80}
        )
        return scorer

    @pytest.fixture
    def pipeline(self, mock_domain_service, mock_ai_service, mock_text_service, 
                 mock_cache_service, mock_confidence_scorer):
        """Provides an ExtractionPipeline instance with mocked dependencies."""
        return ExtractionPipeline(
            domain_service=mock_domain_service,
            ai_service=mock_ai_service,
            text_service=mock_text_service,
            cache_service=mock_cache_service,
            confidence_scorer=mock_confidence_scorer
        )

    @pytest.fixture
    def sample_options(self):
        """Provides sample ExtractionOptions."""
        return ExtractionOptions(
            domain_override="invoice",
            selected_fields=["invoice_number", "total_amount"],
            custom_instructions="Extract invoice details",
            enrich=False,
            use_ocr=False
        )

    def test_init(self, mock_domain_service, mock_ai_service, mock_text_service,
                   mock_cache_service, mock_confidence_scorer):
        """Test ExtractionPipeline initialization."""
        pipeline = ExtractionPipeline(
            domain_service=mock_domain_service,
            ai_service=mock_ai_service,
            text_service=mock_text_service,
            cache_service=mock_cache_service,
            confidence_scorer=mock_confidence_scorer
        )
        
        assert pipeline.domain_service == mock_domain_service
        assert pipeline.ai_service == mock_ai_service
        assert pipeline.text_service == mock_text_service
        assert pipeline.cache_service == mock_cache_service
        assert pipeline.confidence_scorer == mock_confidence_scorer
        assert pipeline.plugin_registry is not None

    def test_extract_success(self, pipeline, sample_options):
        """Test successful extraction flow."""
        file_path = "/test/path/invoice.pdf"
        
        result = pipeline.extract(file_path, sample_options)
        
        # Verify result type
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.INVOICE
        assert result.pages == 1
        
        # Verify service calls
        pipeline.cache_service.get_cached_result.assert_called_once_with(file_path, sample_options)
        pipeline.text_service.extract_text.assert_called_once_with(file_path, False)
        pipeline.domain_service.normalize_domain.assert_called_once_with("invoice")
        pipeline.domain_service.validate_selected_fields.assert_called_once_with("invoice", ["invoice_number", "total_amount"])
        pipeline.ai_service.ai_only_extract.assert_called_once()
        pipeline.confidence_scorer.score.assert_called_once()
        pipeline.cache_service.cache_result.assert_called_once()

    def test_extract_with_cached_result(self, pipeline, sample_options):
        """Test extraction with cached result."""
        file_path = "/test/path/invoice.pdf"
        cached_result = ExtractionResult(
            type=DocumentType.INVOICE,
            pages=1,
            entities=ExtractedEntities(),
            mapped_fields=MappedFields(),
            validation=ValidationResult(),
            confidence=ConfidenceScore(overall=85),
            tables=[],
            formulas=[],
            provenance=Provenance()
        )
        pipeline.cache_service.get_cached_result.return_value = cached_result
        
        result = pipeline.extract(file_path, sample_options)
        
        assert result == cached_result
        # Should not call other services when cache hit
        pipeline.text_service.extract_text.assert_not_called()
        pipeline.ai_service.ai_only_extract.assert_not_called()

    def test_extract_with_enrichment(self, pipeline, sample_options):
        """Test extraction with enrichment enabled."""
        sample_options.enrich = True
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        # Verify enrichment was called
        pipeline.ai_service.enrich_extraction_with_llm.assert_called_once()
        
        # Verify result contains enriched data
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.INVOICE

    def test_extract_with_ocr(self, pipeline, sample_options):
        """Test extraction with OCR enabled."""
        sample_options.use_ocr = True
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        # Verify OCR was requested
        pipeline.text_service.extract_text.assert_called_once_with("/test/path/invoice.pdf", True)
        
        assert isinstance(result, ExtractionResult)

    def test_extract_with_custom_fields(self, pipeline, sample_options):
        """Test extraction with custom fields."""
        # Mock AI service to return custom fields
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "invoice",
            "entities": {},
            "mapped_fields": {"invoice_number": "INV-001"},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {"custom_field": "custom_value"},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.custom_fields is not None
        assert result.custom_fields.custom_field == "custom_value"

    def test_extract_with_plugin_execution(self, pipeline, sample_options):
        """Test extraction with plugin execution."""
        # Mock plugin registry with plugins
        mock_table_plugin = Mock()
        mock_table_plugin.extract.return_value = [{"page": 1, "data": "table_data"}]
        
        mock_formula_plugin = Mock()
        mock_formula_plugin.extract.return_value = ["formula1", "formula2"]
        
        pipeline.plugin_registry.plugins = {
            "table_extractor": mock_table_plugin,
            "formula_extractor": mock_formula_plugin
        }
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        # Verify plugins were called
        mock_table_plugin.extract.assert_called_once()
        mock_formula_plugin.extract.assert_called_once()
        
        # Verify result contains plugin data
        assert isinstance(result, ExtractionResult)
        assert len(result.tables) == 1
        assert len(result.formulas) == 2

    def test_extract_error_handling(self, pipeline, sample_options):
        """Test extraction error handling."""
        # Mock text service to raise exception
        pipeline.text_service.extract_text.side_effect = Exception("Text extraction failed")
        
        with pytest.raises(Exception, match="Text extraction failed"):
            pipeline.extract("/test/path/invoice.pdf", sample_options)

    def test_extract_ai_service_error(self, pipeline, sample_options):
        """Test extraction with AI service error."""
        # Mock AI service to raise exception
        pipeline.ai_service.ai_only_extract.side_effect = Exception("AI service failed")
        
        with pytest.raises(Exception, match="AI service failed"):
            pipeline.extract("/test/path/invoice.pdf", sample_options)

    def test_extract_confidence_scorer_error(self, pipeline, sample_options):
        """Test extraction with confidence scorer error."""
        # Mock confidence scorer to raise exception
        pipeline.confidence_scorer.score.side_effect = Exception("Confidence scoring failed")
        
        with pytest.raises(Exception, match="Confidence scoring failed"):
            pipeline.extract("/test/path/invoice.pdf", sample_options)

    def test_extract_with_empty_selected_fields(self, pipeline, sample_options):
        """Test extraction with empty selected fields."""
        sample_options.selected_fields = []
        pipeline.domain_service.validate_selected_fields.return_value = []
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        pipeline.domain_service.validate_selected_fields.assert_called_once_with("invoice", [])

    def test_extract_with_none_domain_override(self, pipeline, sample_options):
        """Test extraction with None domain override."""
        sample_options.domain_override = None
        pipeline.domain_service.normalize_domain.return_value = "general"
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        pipeline.domain_service.normalize_domain.assert_called_once_with(None)

    def test_extract_with_empty_custom_instructions(self, pipeline, sample_options):
        """Test extraction with empty custom instructions."""
        sample_options.custom_instructions = ""
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        # Verify AI service was called with empty instructions
        call_args = pipeline.ai_service.ai_only_extract.call_args
        assert call_args[0][2] == ""  # custom_instructions parameter

    def test_extract_with_large_text(self, pipeline, sample_options):
        """Test extraction with large text content."""
        large_text = "x" * 50000
        pipeline.text_service.extract_text.return_value = (
            large_text,
            [large_text],
            {"test": [0]}
        )
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        # Verify AI service was called with chunked text
        call_args = pipeline.ai_service.ai_only_extract.call_args
        ai_text = call_args[0][0]  # all_text parameter
        assert len(ai_text) < len(large_text)  # Should be chunked

    def test_extract_with_validation_errors(self, pipeline, sample_options):
        """Test extraction with validation errors."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "invoice",
            "entities": {},
            "mapped_fields": {},
            "validation": {
                "errors": ["Invalid invoice number format"],
                "warnings": ["Missing due date"]
            },
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.validation.errors) == 1
        assert len(result.validation.warnings) == 1
        assert "Invalid invoice number format" in result.validation.errors
        assert "Missing due date" in result.validation.warnings

    def test_extract_with_research_document(self, pipeline, sample_options):
        """Test extraction with research document type."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "research",
            "entities": {},
            "mapped_fields": {
                "title": "Research Paper",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Paper abstract"
            },
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        pipeline.domain_service.normalize_domain.return_value = "research"
        
        result = pipeline.extract("/test/path/research.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.RESEARCH
        assert result.mapped_fields.title == "Research Paper"
        assert len(result.mapped_fields.authors) == 2

    def test_extract_with_healthcare_document(self, pipeline, sample_options):
        """Test extraction with healthcare document type."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "healthcare",
            "entities": {},
            "mapped_fields": {
                "patient_name": "John Doe",
                "diagnosis": "Hypertension",
                "medications": ["Medication A", "Medication B"]
            },
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        pipeline.domain_service.normalize_domain.return_value = "healthcare"
        
        result = pipeline.extract("/test/path/medical.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.HEALTHCARE
        assert result.mapped_fields.patient_name == "John Doe"
        assert result.mapped_fields.diagnosis == "Hypertension"

    def test_extract_with_financial_document(self, pipeline, sample_options):
        """Test extraction with financial document type."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "financial",
            "entities": {},
            "mapped_fields": {
                "company_name": "ACME Corp",
                "revenue": "$1,000,000",
                "profit_margin": "15%"
            },
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        pipeline.domain_service.normalize_domain.return_value = "financial"
        
        result = pipeline.extract("/test/path/financial.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.FINANCIAL
        assert result.mapped_fields.company_name == "ACME Corp"
        assert result.mapped_fields.revenue == "$1,000,000"

    def test_extract_with_contract_document(self, pipeline, sample_options):
        """Test extraction with contract document type."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "contract",
            "entities": {},
            "mapped_fields": {
                "contract_type": "Service Agreement",
                "parties": ["Party A", "Party B"],
                "effective_date": "2024-01-01",
                "termination_date": "2024-12-31"
            },
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        pipeline.domain_service.normalize_domain.return_value = "contract"
        
        result = pipeline.extract("/test/path/contract.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.CONTRACT
        assert result.mapped_fields.contract_type == "Service Agreement"
        assert len(result.mapped_fields.parties) == 2

    def test_extract_with_general_document(self, pipeline, sample_options):
        """Test extraction with general document type."""
        pipeline.ai_service.ai_only_extract.return_value = {
            "type": "general",
            "entities": {
                "emails": ["test@example.com"],
                "phones": ["+1234567890"],
                "amounts": ["$100.00"],
                "dates": ["2024-01-01"]
            },
            "mapped_fields": {},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        pipeline.domain_service.normalize_domain.return_value = "general"
        
        result = pipeline.extract("/test/path/general.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.type == DocumentType.GENERAL
        assert len(result.entities.emails) == 1
        assert len(result.entities.phones) == 1
        assert len(result.entities.amounts) == 1
        assert len(result.entities.dates) == 1

    def test_extract_cache_key_generation(self, pipeline, sample_options):
        """Test that cache key is properly generated."""
        file_path = "/test/path/invoice.pdf"
        
        pipeline.extract(file_path, sample_options)
        
        # Verify cache service was called with correct parameters
        pipeline.cache_service.get_cached_result.assert_called_once_with(file_path, sample_options)
        pipeline.cache_service.cache_result.assert_called_once()
        
        # Verify cache_result was called with ExtractionResult
        cache_call_args = pipeline.cache_service.cache_result.call_args
        assert len(cache_call_args[0]) == 3  # file_path, options, result
        assert isinstance(cache_call_args[0][2], ExtractionResult)

    def test_extract_plugin_registry_initialization(self, pipeline):
        """Test that plugin registry is properly initialized."""
        assert pipeline.plugin_registry is not None
        assert hasattr(pipeline.plugin_registry, 'plugins')
        assert isinstance(pipeline.plugin_registry.plugins, dict)

    def test_extract_with_no_plugins(self, pipeline, sample_options):
        """Test extraction with no plugins registered."""
        pipeline.plugin_registry.plugins = {}
        
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.tables == []
        assert result.formulas == []

    def test_extract_with_plugin_error(self, pipeline, sample_options):
        """Test extraction with plugin error."""
        mock_plugin = Mock()
        mock_plugin.extract.side_effect = Exception("Plugin error")
        
        pipeline.plugin_registry.plugins = {"table_extractor": mock_plugin}
        
        # Should not raise exception, just continue without plugin results
        result = pipeline.extract("/test/path/invoice.pdf", sample_options)
        
        assert isinstance(result, ExtractionResult)
        assert result.tables == []
        assert result.formulas == []
