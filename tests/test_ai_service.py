# SPDX-License-Identifier: AGPL-3.0-only

import pytest
from unittest.mock import Mock, patch, MagicMock
from extraction.ai_service import AIExtractionService
from extraction.domain_service import DomainService
from extraction.models import DocumentType


class TestAIExtractionService:
    """Test suite for AIExtractionService."""

    @pytest.fixture
    def domain_service(self):
        """Provides a DomainService instance for tests."""
        return DomainService()

    @pytest.fixture
    def ai_service(self, domain_service):
        """Provides an AIExtractionService instance for tests."""
        return AIExtractionService(domain_service)

    def test_init(self, domain_service):
        """Test AIExtractionService initialization."""
        service = AIExtractionService(domain_service)
        assert service.domain_service == domain_service
        assert service.config is not None

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_success(self, mock_call_ai, ai_service):
        """Test successful AI-only extraction."""
        # Mock AI response
        mock_response = {
            "type": "invoice",
            "entities": {
                "emails": ["test@example.com"],
                "phones": ["+1234567890"],
                "amounts": ["$100.00"],
                "dates": ["2024-01-01"]
            },
            "mapped_fields": {
                "invoice_number": "INV-001",
                "total_amount": "$100.00",
                "due_date": "2024-01-15"
            },
            "validation": {
                "errors": [],
                "warnings": []
            },
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        mock_call_ai.return_value = mock_response

        # Test data
        all_text = "Invoice INV-001 Total: $100.00 Due: 2024-01-15"
        pages_text = [all_text]
        custom_instructions = "Extract invoice details"
        domain = "invoice"
        selected_fields = ["invoice_number", "total_amount", "due_date"]

        # Execute
        result = ai_service.ai_only_extract(
            all_text, pages_text, custom_instructions, domain, selected_fields
        )

        # Verify
        assert result["type"] == "invoice"
        assert result["entities"]["emails"] == ["test@example.com"]
        assert result["mapped_fields"]["invoice_number"] == "INV-001"
        assert result["validation"]["errors"] == []
        mock_call_ai.assert_called_once()

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_with_custom_instructions(self, mock_call_ai, ai_service):
        """Test AI extraction with custom instructions."""
        mock_response = {
            "type": "general",
            "entities": {},
            "mapped_fields": {},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {},
            "custom_instructions_result": "Custom analysis completed"
        }
        mock_call_ai.return_value = mock_response

        result = ai_service.ai_only_extract(
            "Test document", ["Test document"], "Analyze sentiment", "general", []
        )

        assert result["custom_instructions_result"] == "Custom analysis completed"
        mock_call_ai.assert_called_once()

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_error_handling(self, mock_call_ai, ai_service):
        """Test AI extraction error handling."""
        mock_call_ai.side_effect = Exception("AI service error")

        result = ai_service.ai_only_extract(
            "Test document", ["Test document"], "", "general", []
        )

        # Should return error response
        assert "error" in result
        assert result["type"] == "general"
        assert result["entities"] == {}
        assert result["mapped_fields"] == {}

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_invalid_json(self, mock_call_ai, ai_service):
        """Test AI extraction with invalid JSON response."""
        mock_call_ai.return_value = "Invalid JSON response"

        result = ai_service.ai_only_extract(
            "Test document", ["Test document"], "", "general", []
        )

        # Should handle invalid JSON gracefully
        assert result["type"] == "general"
        assert result["entities"] == {}
        assert result["mapped_fields"] == {}

    @patch('extraction.ai_service.call_ai_service')
    def test_enrich_extraction_with_llm_success(self, mock_call_ai, ai_service):
        """Test successful LLM enrichment."""
        mock_response = {
            "mapped_fields": {
                "enrichment": {
                    "key_findings": ["Finding 1", "Finding 2"],
                    "summary": "Document summary"
                }
            }
        }
        mock_call_ai.return_value = mock_response

        # Test data
        dtype = DocumentType.RESEARCH
        mapped_fields = {"title": "Research Paper"}
        all_text = "Research paper content"
        pages_text = ["Page 1 content", "Page 2 content"]
        inv_index = {"research": [0, 1]}

        result = ai_service.enrich_extraction_with_llm(
            dtype, mapped_fields, all_text, pages_text, inv_index
        )

        assert result["mapped_fields"]["enrichment"]["key_findings"] == ["Finding 1", "Finding 2"]
        mock_call_ai.assert_called_once()

    @patch('extraction.ai_service.call_ai_service')
    def test_enrich_extraction_with_llm_error(self, mock_call_ai, ai_service):
        """Test LLM enrichment error handling."""
        mock_call_ai.side_effect = Exception("Enrichment error")

        result = ai_service.enrich_extraction_with_llm(
            DocumentType.RESEARCH, {}, "text", ["page"], {}
        )

        # Should return None on error
        assert result is None

    @patch('extraction.ai_service.call_ai_service')
    def test_enrich_extraction_with_llm_invalid_response(self, mock_call_ai, ai_service):
        """Test LLM enrichment with invalid response."""
        mock_call_ai.return_value = "Invalid response"

        result = ai_service.enrich_extraction_with_llm(
            DocumentType.RESEARCH, {}, "text", ["page"], {}
        )

        # Should return None for invalid response
        assert result is None

    def test_build_prompt_with_domain(self, ai_service):
        """Test prompt building with specific domain."""
        domain = "invoice"
        selected_fields = ["invoice_number", "total_amount"]
        
        prompt = ai_service._build_prompt(
            "Test invoice", domain, selected_fields, "Custom instructions"
        )
        
        assert "invoice" in prompt.lower()
        assert "invoice_number" in prompt
        assert "total_amount" in prompt
        assert "Custom instructions" in prompt

    def test_build_prompt_general_domain(self, ai_service):
        """Test prompt building with general domain."""
        prompt = ai_service._build_prompt(
            "Test document", "general", [], ""
        )
        
        assert "general" in prompt.lower()
        assert "detect" in prompt.lower()

    def test_build_prompt_text_chunking(self, ai_service):
        """Test prompt building with text chunking for large documents."""
        large_text = "x" * 25000  # Exceeds max_chars
        
        prompt = ai_service._build_prompt(
            large_text, "general", [], ""
        )
        
        # Should contain chunked text markers
        assert "[MIDDLE]" in prompt
        assert "[END]" in prompt
        assert len(prompt) < len(large_text)

    @patch('extraction.ai_service.call_ai_service')
    def test_call_ai_with_timeout(self, mock_call_ai, ai_service):
        """Test AI call with timeout handling."""
        mock_call_ai.side_effect = Exception("Timeout")
        
        result = ai_service._call_ai("test prompt")
        
        # Should handle timeout gracefully
        assert result is None

    @patch('extraction.ai_service.call_ai_service')
    def test_call_ai_success(self, mock_call_ai, ai_service):
        """Test successful AI call."""
        mock_response = {"test": "response"}
        mock_call_ai.return_value = mock_response
        
        result = ai_service._call_ai("test prompt")
        
        assert result == mock_response
        mock_call_ai.assert_called_once()

    def test_parse_json_safely_valid_json(self, ai_service):
        """Test JSON parsing with valid JSON."""
        valid_json = '{"key": "value", "number": 123}'
        
        result = ai_service.parse_json_safely(valid_json)
        
        assert result == {"key": "value", "number": 123}

    def test_parse_json_safely_invalid_json(self, ai_service):
        """Test JSON parsing with invalid JSON."""
        invalid_json = '{"key": "value", "number": 123'  # Missing closing brace
        
        result = ai_service.parse_json_safely(invalid_json)
        
        assert result is None

    def test_parse_json_safely_code_fences(self, ai_service):
        """Test JSON parsing with code fences."""
        json_with_fences = '```json\n{"key": "value"}\n```'
        
        result = ai_service.parse_json_safely(json_with_fences)
        
        assert result == {"key": "value"}

    def test_parse_json_safely_empty_string(self, ai_service):
        """Test JSON parsing with empty string."""
        result = ai_service.parse_json_safely("")
        
        assert result is None

    def test_parse_json_safely_none_input(self, ai_service):
        """Test JSON parsing with None input."""
        result = ai_service.parse_json_safely(None)
        
        assert result is None

    def test_strip_code_fences(self, ai_service):
        """Test code fence stripping."""
        text_with_fences = '```json\n{"key": "value"}\n```'
        
        result = ai_service._strip_code_fences(text_with_fences)
        
        assert result == '{"key": "value"}'

    def test_strip_code_fences_no_fences(self, ai_service):
        """Test code fence stripping with no fences."""
        text = '{"key": "value"}'
        
        result = ai_service._strip_code_fences(text)
        
        assert result == text

    def test_object_to_readable(self, ai_service):
        """Test object to readable string conversion."""
        test_obj = {"key": "value", "nested": {"inner": "data"}}
        
        result = ai_service._object_to_readable(test_obj)
        
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_object_to_readable_list(self, ai_service):
        """Test object to readable string conversion with list."""
        test_obj = ["item1", "item2", {"nested": "value"}]
        
        result = ai_service._object_to_readable(test_obj)
        
        assert isinstance(result, str)
        assert "item1" in result
        assert "item2" in result

    def test_object_to_readable_string(self, ai_service):
        """Test object to readable string conversion with string."""
        test_obj = "simple string"
        
        result = ai_service._object_to_readable(test_obj)
        
        assert result == "simple string"

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_with_selected_fields(self, mock_call_ai, ai_service):
        """Test AI extraction with selected fields filtering."""
        mock_response = {
            "type": "invoice",
            "entities": {"emails": ["test@example.com"]},
            "mapped_fields": {
                "invoice_number": "INV-001",
                "total_amount": "$100.00",
                "unwanted_field": "should_not_be_here"
            },
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        mock_call_ai.return_value = mock_response

        selected_fields = ["invoice_number", "total_amount"]
        
        result = ai_service.ai_only_extract(
            "Invoice content", ["Invoice content"], "", "invoice", selected_fields
        )

        # The AI service should respect selected fields in the prompt
        # The actual filtering happens in the prompt construction
        assert result["type"] == "invoice"
        mock_call_ai.assert_called_once()

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_domain_validation(self, mock_call_ai, ai_service):
        """Test AI extraction with domain validation."""
        mock_response = {
            "type": "contract",
            "entities": {},
            "mapped_fields": {},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        mock_call_ai.return_value = mock_response

        result = ai_service.ai_only_extract(
            "Contract content", ["Contract content"], "", "contract", []
        )

        assert result["type"] == "contract"
        mock_call_ai.assert_called_once()

    def test_config_loading(self, ai_service):
        """Test that configuration is properly loaded."""
        assert ai_service.config is not None
        assert hasattr(ai_service.config, 'AI_SERVICE')
        assert hasattr(ai_service.config, 'AI_TIMEOUT')
        assert hasattr(ai_service.config, 'MAX_TEXT_LENGTH')

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_with_empty_text(self, mock_call_ai, ai_service):
        """Test AI extraction with empty text."""
        mock_response = {
            "type": "general",
            "entities": {},
            "mapped_fields": {},
            "validation": {"errors": [], "warnings": []},
            "custom_fields": {},
            "tables": [],
            "formulas": [],
            "provenance": {}
        }
        mock_call_ai.return_value = mock_response

        result = ai_service.ai_only_extract(
            "", [], "", "general", []
        )

        assert result["type"] == "general"
        assert result["entities"] == {}
        assert result["mapped_fields"] == {}
        mock_call_ai.assert_called_once()

    @patch('extraction.ai_service.call_ai_service')
    def test_ai_only_extract_with_none_values(self, mock_call_ai, ai_service):
        """Test AI extraction with None values in response."""
        mock_response = {
            "type": "general",
            "entities": None,
            "mapped_fields": None,
            "validation": None,
            "custom_fields": None,
            "tables": None,
            "formulas": None,
            "provenance": None
        }
        mock_call_ai.return_value = mock_response

        result = ai_service.ai_only_extract(
            "Test", ["Test"], "", "general", []
        )

        # Should handle None values gracefully
        assert result["type"] == "general"
        assert result["entities"] == {}
        assert result["mapped_fields"] == {}
        assert result["validation"] == {"errors": [], "warnings": []}
        assert result["custom_fields"] == {}
        assert result["tables"] == []
        assert result["formulas"] == []
        assert result["provenance"] == {}
