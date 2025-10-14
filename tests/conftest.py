# SPDX-License-Identifier: AGPL-3.0-only

"""
Pytest configuration and fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, MagicMock

from extraction.models import ExtractionOptions, ExtractionResult, ValidationResult, ConfidenceScore
from extraction.domain_service import DomainService
from extraction.ai_service import AIExtractionService
from extraction.text_service import TextExtractionService
from extraction.cache_service import CacheService
from extraction.confidence import ConfidenceScorer


@pytest.fixture
def sample_pdf_path():
    """Create a temporary PDF file for testing."""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_file.close()
    
    # Write some dummy content (not a real PDF, but sufficient for testing)
    with open(temp_file.name, 'wb') as f:
        f.write(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n')
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    INVOICE
    
    Invoice Number: INV-2024-001
    Date: 2024-01-15
    Due Date: 2024-02-15
    
    Vendor: Acme Corp
    Address: 123 Main St, City, State 12345
    Email: vendor@acme.com
    Phone: (555) 123-4567
    
    Customer: Customer Inc
    Address: 456 Oak Ave, City, State 67890
    Email: customer@customer.com
    Phone: (555) 987-6543
    
    Items:
    - Product A: $100.00
    - Product B: $200.00
    
    Subtotal: $300.00
    Tax (10%): $30.00
    Total: $330.00
    
    Payment Terms: Net 30
    """


@pytest.fixture
def sample_pages_text():
    """Sample pages text for testing."""
    return [
        "INVOICE\n\nInvoice Number: INV-2024-001\nDate: 2024-01-15",
        "Vendor: Acme Corp\nAddress: 123 Main St\nEmail: vendor@acme.com",
        "Items:\n- Product A: $100.00\n- Product B: $200.00",
        "Subtotal: $300.00\nTax (10%): $30.00\nTotal: $330.00"
    ]


@pytest.fixture
def extraction_options():
    """Sample extraction options."""
    return ExtractionOptions(
        domain_override="invoice",
        selected_fields=["invoice_number", "total_amount", "vendor_name"],
        custom_instructions="Extract dates in ISO format",
        enrich=False,
        use_ocr=False
    )


@pytest.fixture
def extraction_result():
    """Sample extraction result."""
    return ExtractionResult(
        type="invoice",
        pages=4,
        entities={
            "emails": ["vendor@acme.com", "customer@customer.com"],
            "phones": ["(555) 123-4567", "(555) 987-6543"],
            "amounts": ["$100.00", "$200.00", "$300.00", "$30.00", "$330.00"],
            "dates": ["2024-01-15", "2024-02-15"]
        },
        mapped_fields={
            "invoice_number": "INV-2024-001",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
            "vendor_name": "Acme Corp",
            "customer_name": "Customer Inc",
            "total_amount": "$330.00"
        },
        validation=ValidationResult(),
        confidence=ConfidenceScore(overall=85, fields={
            "invoice_number": 90,
            "total_amount": 95,
            "vendor_name": 80
        }),
        tables=[],
        formulas=[],
        provenance=None
    )


@pytest.fixture
def mock_ocr_service():
    """Mock OCR service."""
    mock_service = Mock()
    mock_service.extract_text.return_value = {
        'text': 'Sample OCR text',
        'pages': ['Page 1 text', 'Page 2 text']
    }
    return mock_service


@pytest.fixture
def domain_service():
    """Domain service instance."""
    return DomainService()


@pytest.fixture
def mock_ai_service(domain_service):
    """Mock AI service."""
    mock_service = Mock(spec=AIExtractionService)
    mock_service.extract_document.return_value = ExtractionResult(
        type="invoice",
        pages=1,
        entities={},
        mapped_fields={"invoice_number": "INV-001"},
        validation=ValidationResult(),
        confidence=ConfidenceScore(overall=80)
    )
    mock_service.enrich_extraction.return_value = {
        'mapped_fields': {"invoice_number": "INV-001", "enriched": True}
    }
    return mock_service


@pytest.fixture
def mock_text_service(mock_ocr_service):
    """Mock text service."""
    mock_service = Mock(spec=TextExtractionService)
    mock_service.extract_text_from_pdf.return_value = (
        "Sample extracted text",
        ["Page 1", "Page 2"]
    )
    mock_service.build_inverted_index.return_value = {
        "sample": [1, 2],
        "text": [1, 2]
    }
    return mock_service


@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    mock_service = Mock(spec=CacheService)
    mock_service.get_cached_result.return_value = None
    mock_service.cache_result.return_value = None
    mock_service.compute_cache_key.return_value = "test_cache_key"
    mock_service.get_cache_stats.return_value = {
        'memory_entries': 0,
        'disk_entries': 0,
        'total_entries': 0
    }
    return mock_service


@pytest.fixture
def confidence_scorer():
    """Confidence scorer instance."""
    return ConfidenceScorer()


@pytest.fixture
def mock_pipeline(domain_service, mock_ai_service, mock_text_service, mock_cache_service, confidence_scorer):
    """Mock extraction pipeline."""
    mock_pipeline = Mock()
    mock_pipeline.extract.return_value = ExtractionResult(
        type="invoice",
        pages=1,
        entities={},
        mapped_fields={"invoice_number": "INV-001"},
        validation=ValidationResult(),
        confidence=ConfidenceScore(overall=80)
    )
    return mock_pipeline


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture
def mock_file_upload():
    """Mock file upload object."""
    mock_file = Mock()
    mock_file.filename = "test.pdf"
    mock_file.read.return_value = b"PDF content"
    mock_file.save.return_value = None
    return mock_file


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test names
    for item in items:
        if "test_" in item.name:
            if "integration" in item.name:
                item.add_marker(pytest.mark.integration)
            elif "unit" in item.name:
                item.add_marker(pytest.mark.unit)
            else:
                item.add_marker(pytest.mark.unit)  # Default to unit test
