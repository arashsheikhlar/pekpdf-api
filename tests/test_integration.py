# SPDX-License-Identifier: AGPL-3.0-only

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from extraction.pipeline import ExtractionPipeline
from extraction.domain_service import DomainService
from extraction.ai_service import AIExtractionService
from extraction.text_service import TextExtractionService
from extraction.cache_service import CacheService
from extraction.confidence import ConfidenceScorer
from extraction.models import ExtractionOptions, DocumentType
from extraction.ocr_service import OCRService


class TestIntegration:
    """Integration tests for the extraction system."""

    @pytest.fixture
    def sample_pdf_content(self):
        """Provides sample PDF content for testing."""
        return """
        INVOICE
        
        Invoice Number: INV-001
        Date: 2024-01-15
        Due Date: 2024-02-15
        
        Bill To:
        ACME Corporation
        123 Business St
        City, State 12345
        
        Description: Professional Services
        Amount: $1,000.00
        
        Total: $1,000.00
        
        Contact: billing@acme.com
        Phone: (555) 123-4567
        """

    @pytest.fixture
    def sample_research_content(self):
        """Provides sample research paper content for testing."""
        return """
        Title: Machine Learning Applications in Healthcare
        
        Authors: Dr. Jane Smith, Dr. John Doe
        
        Abstract:
        This paper presents a comprehensive study of machine learning applications
        in healthcare, focusing on diagnostic accuracy and patient outcomes.
        
        Introduction:
        Machine learning has revolutionized healthcare by enabling more accurate
        diagnoses and personalized treatment plans.
        
        Methods:
        We conducted a systematic review of 100 studies published between 2020-2024.
        
        Results:
        Our analysis shows a 25% improvement in diagnostic accuracy with ML systems.
        
        Conclusions:
        Machine learning significantly improves healthcare outcomes.
        
        References:
        1. Smith, J. (2023). AI in Healthcare. Journal of Medical AI.
        2. Doe, J. (2024). Machine Learning Applications. Healthcare Tech Review.
        """

    @pytest.fixture
    def sample_contract_content(self):
        """Provides sample contract content for testing."""
        return """
        SERVICE AGREEMENT
        
        This Service Agreement ("Agreement") is entered into on January 1, 2024,
        between ACME Corporation ("Client") and Tech Solutions Inc. ("Provider").
        
        Term: This agreement shall commence on January 1, 2024, and continue
        until December 31, 2024, unless terminated earlier.
        
        Services: Provider shall deliver software development services including
        web application development, database design, and system integration.
        
        Payment: Client shall pay Provider $10,000 per month for services rendered.
        
        Termination: Either party may terminate this agreement with 30 days notice.
        
        Contact Information:
        Client: legal@acme.com, (555) 123-4567
        Provider: contracts@techsolutions.com, (555) 987-6543
        """

    @pytest.fixture
    def sample_healthcare_content(self):
        """Provides sample healthcare document content for testing."""
        return """
        MEDICAL REPORT
        
        Patient: John Doe
        DOB: 1980-05-15
        Patient ID: 12345
        
        Chief Complaint: Chest pain
        
        History of Present Illness:
        Patient presents with acute chest pain that started 2 hours ago.
        Pain is described as sharp and localized to the left side.
        
        Assessment:
        - Hypertension
        - Hyperlipidemia
        - Chest pain, likely musculoskeletal
        
        Plan:
        1. EKG to rule out cardiac event
        2. Chest X-ray
        3. Blood work including troponin levels
        4. Prescribe ibuprofen 400mg TID for pain
        
        Medications:
        - Lisinopril 10mg daily
        - Atorvastatin 20mg daily
        - Ibuprofen 400mg TID as needed
        
        Follow-up: Return in 1 week if symptoms persist
        
        Provider: Dr. Sarah Johnson
        Date: 2024-01-15
        """

    @pytest.fixture
    def sample_financial_content(self):
        """Provides sample financial document content for testing."""
        return """
        QUARTERLY FINANCIAL REPORT
        
        Company: TechCorp Inc.
        Period: Q4 2023
        Report Date: January 15, 2024
        
        Revenue: $2,500,000
        Cost of Goods Sold: $1,200,000
        Gross Profit: $1,300,000
        
        Operating Expenses:
        - Salaries: $800,000
        - Marketing: $150,000
        - Rent: $100,000
        - Utilities: $50,000
        
        Operating Income: $200,000
        
        Net Income: $180,000
        Profit Margin: 7.2%
        
        Key Metrics:
        - Revenue Growth: 15% YoY
        - Customer Acquisition Cost: $150
        - Customer Lifetime Value: $2,500
        
        Contact: finance@techcorp.com
        Phone: (555) 555-0123
        """

    @pytest.fixture
    def mock_ocr_service(self):
        """Provides a mocked OCR service."""
        service = Mock(spec=OCRService)
        service.hybrid_extraction.return_value = {
            'success': True,
            'pages': [{'text': 'Sample OCR text'}]
        }
        return service

    @pytest.fixture
    def real_services(self, mock_ocr_service):
        """Provides real service instances for integration testing."""
        domain_service = DomainService()
        ai_service = AIExtractionService(domain_service)
        text_service = TextExtractionService(mock_ocr_service)
        cache_service = CacheService()
        confidence_scorer = ConfidenceScorer()
        
        return {
            'domain_service': domain_service,
            'ai_service': ai_service,
            'text_service': text_service,
            'cache_service': cache_service,
            'confidence_scorer': confidence_scorer
        }

    @pytest.fixture
    def pipeline(self, real_services):
        """Provides a real ExtractionPipeline instance."""
        return ExtractionPipeline(
            domain_service=real_services['domain_service'],
            ai_service=real_services['ai_service'],
            text_service=real_services['text_service'],
            cache_service=real_services['cache_service'],
            confidence_scorer=real_services['confidence_scorer']
        )

    def create_temp_pdf(self, content: str) -> str:
        """Create a temporary PDF file with given content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write(content)
            return f.name

    def test_invoice_extraction_integration(self, pipeline, sample_pdf_content):
        """Test end-to-end invoice extraction."""
        # Create temporary PDF file
        pdf_path = self.create_temp_pdf(sample_pdf_content)
        
        try:
            options = ExtractionOptions(
                domain_override="invoice",
                selected_fields=["invoice_number", "total_amount", "due_date", "bill_to"],
                custom_instructions="Extract invoice details accurately",
                enrich=False,
                use_ocr=False
            )
            
            # Mock the PDF reader to return our sample content
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_pdf_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            # Verify result structure
            assert isinstance(result, type(pipeline.extract.__annotations__['return']))
            assert result.type == DocumentType.INVOICE
            assert result.pages == 1
            
            # Verify extracted data (these would be populated by AI service in real scenario)
            assert result.entities is not None
            assert result.mapped_fields is not None
            assert result.validation is not None
            assert result.confidence is not None
            
        finally:
            # Clean up
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_research_extraction_integration(self, pipeline, sample_research_content):
        """Test end-to-end research paper extraction."""
        pdf_path = self.create_temp_pdf(sample_research_content)
        
        try:
            options = ExtractionOptions(
                domain_override="research",
                selected_fields=["title", "authors", "abstract", "conclusions"],
                custom_instructions="Extract research paper metadata",
                enrich=True,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_research_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            assert result.type == DocumentType.RESEARCH
            assert result.pages == 1
            assert result.entities is not None
            assert result.mapped_fields is not None
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_contract_extraction_integration(self, pipeline, sample_contract_content):
        """Test end-to-end contract extraction."""
        pdf_path = self.create_temp_pdf(sample_contract_content)
        
        try:
            options = ExtractionOptions(
                domain_override="contract",
                selected_fields=["contract_type", "parties", "effective_date", "termination_date"],
                custom_instructions="Extract contract terms and parties",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_contract_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            assert result.type == DocumentType.CONTRACT
            assert result.pages == 1
            assert result.entities is not None
            assert result.mapped_fields is not None
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_healthcare_extraction_integration(self, pipeline, sample_healthcare_content):
        """Test end-to-end healthcare document extraction."""
        pdf_path = self.create_temp_pdf(sample_healthcare_content)
        
        try:
            options = ExtractionOptions(
                domain_override="healthcare",
                selected_fields=["patient_name", "diagnosis", "medications", "provider"],
                custom_instructions="Extract medical information carefully",
                enrich=True,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_healthcare_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            assert result.type == DocumentType.HEALTHCARE
            assert result.pages == 1
            assert result.entities is not None
            assert result.mapped_fields is not None
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_financial_extraction_integration(self, pipeline, sample_financial_content):
        """Test end-to-end financial document extraction."""
        pdf_path = self.create_temp_pdf(sample_financial_content)
        
        try:
            options = ExtractionOptions(
                domain_override="financial",
                selected_fields=["company_name", "revenue", "profit_margin", "period"],
                custom_instructions="Extract financial metrics accurately",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_financial_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            assert result.type == DocumentType.FINANCIAL
            assert result.pages == 1
            assert result.entities is not None
            assert result.mapped_fields is not None
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_general_extraction_integration(self, pipeline, sample_pdf_content):
        """Test end-to-end general document extraction."""
        pdf_path = self.create_temp_pdf(sample_pdf_content)
        
        try:
            options = ExtractionOptions(
                domain_override=None,  # Let it auto-detect
                selected_fields=[],
                custom_instructions="",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_pdf_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            assert result.type is not None
            assert result.pages == 1
            assert result.entities is not None
            assert result.mapped_fields is not None
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_caching_integration(self, pipeline, sample_pdf_content):
        """Test caching functionality in integration."""
        pdf_path = self.create_temp_pdf(sample_pdf_content)
        
        try:
            options = ExtractionOptions(
                domain_override="invoice",
                selected_fields=["invoice_number"],
                custom_instructions="",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_pdf_content
                mock_reader.return_value.pages = [mock_page]
                
                # First extraction
                result1 = pipeline.extract(pdf_path, options)
                
                # Second extraction should use cache
                result2 = pipeline.extract(pdf_path, options)
            
            # Both should return valid results
            assert isinstance(result1, type(pipeline.extract.__annotations__['return']))
            assert isinstance(result2, type(pipeline.extract.__annotations__['return']))
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_ocr_integration(self, pipeline, sample_pdf_content, mock_ocr_service):
        """Test OCR functionality in integration."""
        pdf_path = self.create_temp_pdf(sample_pdf_content)
        
        try:
            options = ExtractionOptions(
                domain_override="invoice",
                selected_fields=["invoice_number"],
                custom_instructions="",
                enrich=False,
                use_ocr=True  # Enable OCR
            )
            
            # Mock PDF reader to simulate scanned document (no text)
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = ""  # No text available
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            # Verify OCR was called
            mock_ocr_service.hybrid_extraction.assert_called_once()
            
            # Verify result structure
            assert isinstance(result, type(pipeline.extract.__annotations__['return']))
            assert result.pages == 1
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_plugin_integration(self, pipeline, sample_pdf_content):
        """Test plugin functionality in integration."""
        pdf_path = self.create_temp_pdf(sample_pdf_content)
        
        try:
            # Mock plugins
            mock_table_plugin = Mock()
            mock_table_plugin.extract.return_value = [
                {"page": 1, "rows": 3, "cols": 2, "data": "table_data"}
            ]
            
            mock_formula_plugin = Mock()
            mock_formula_plugin.extract.return_value = [
                "E = mc²",
                "F = ma"
            ]
            
            pipeline.plugin_registry.plugins = {
                "table_extractor": mock_table_plugin,
                "formula_extractor": mock_formula_plugin
            }
            
            options = ExtractionOptions(
                domain_override="invoice",
                selected_fields=["invoice_number"],
                custom_instructions="",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_pdf_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            # Verify plugins were called
            mock_table_plugin.extract.assert_called_once()
            mock_formula_plugin.extract.assert_called_once()
            
            # Verify plugin results are included
            assert len(result.tables) == 1
            assert len(result.formulas) == 2
            assert result.tables[0]["page"] == 1
            assert result.formulas[0] == "E = mc²"
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_error_handling_integration(self, pipeline):
        """Test error handling in integration scenarios."""
        # Test with non-existent file
        with pytest.raises(Exception):
            pipeline.extract("/non/existent/file.pdf", ExtractionOptions())

    def test_large_document_integration(self, pipeline):
        """Test handling of large documents."""
        # Create a large document content
        large_content = "Sample content. " * 10000  # ~150KB of text
        
        pdf_path = self.create_temp_pdf(large_content)
        
        try:
            options = ExtractionOptions(
                domain_override="general",
                selected_fields=[],
                custom_instructions="",
                enrich=False,
                use_ocr=False
            )
            
            with patch('PyPDF2.PdfReader') as mock_reader:
                mock_page = Mock()
                mock_page.extract_text.return_value = large_content
                mock_reader.return_value.pages = [mock_page]
                
                result = pipeline.extract(pdf_path, options)
            
            # Should handle large documents gracefully
            assert isinstance(result, type(pipeline.extract.__annotations__['return']))
            assert result.pages == 1
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_multiple_document_types_integration(self, pipeline):
        """Test extraction of multiple document types in sequence."""
        documents = [
            ("invoice", "Invoice INV-001 Total: $100.00"),
            ("contract", "Service Agreement between Party A and Party B"),
            ("research", "Title: Machine Learning in Healthcare"),
            ("healthcare", "Patient: John Doe Diagnosis: Hypertension"),
            ("financial", "Revenue: $1,000,000 Profit: $200,000")
        ]
        
        for doc_type, content in documents:
            pdf_path = self.create_temp_pdf(content)
            
            try:
                options = ExtractionOptions(
                    domain_override=doc_type,
                    selected_fields=[],
                    custom_instructions="",
                    enrich=False,
                    use_ocr=False
                )
                
                with patch('PyPDF2.PdfReader') as mock_reader:
                    mock_page = Mock()
                    mock_page.extract_text.return_value = content
                    mock_reader.return_value.pages = [mock_page]
                    
                    result = pipeline.extract(pdf_path, options)
                
                # Verify each document type is handled correctly
                assert isinstance(result, type(pipeline.extract.__annotations__['return']))
                assert result.pages == 1
                
            finally:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)

    def test_service_dependency_integration(self, real_services):
        """Test that all services work together correctly."""
        # Test domain service
        assert real_services['domain_service'].normalize_domain("INVOICE") == "invoice"
        assert real_services['domain_service'].validate_selected_fields("invoice", ["invoice_number"]) == ["invoice_number"]
        
        # Test text service
        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test content"
            mock_reader.return_value.pages = [mock_page]
            
            all_text, pages_text, inv_index = real_services['text_service'].extract_text("/test/path", False)
            
            assert all_text == "Test content"
            assert pages_text == ["Test content"]
            assert isinstance(inv_index, dict)
        
        # Test cache service
        assert real_services['cache_service'].get_cached_result("/test/path", ExtractionOptions()) is None
        
        # Test confidence scorer
        confidence = real_services['confidence_scorer'].score(
            "invoice", 
            {"invoice_number": "INV-001"}, 
            {"emails": ["test@example.com"]}, 
            {"errors": [], "warnings": []}, 
            {}, 
            ["invoice_number"]
        )
        
        assert confidence is not None
        assert hasattr(confidence, 'overall')
        assert hasattr(confidence, 'fields')
