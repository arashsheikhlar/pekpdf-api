# Extraction Module Documentation

## Overview

The extraction module provides a modular, scalable architecture for document information extraction from PDF files. It uses AI-powered extraction with optional OCR fallback, caching, and plugin support for extensibility.

## Architecture

The extraction system is built around a pipeline architecture with clear separation of concerns:

```
PDF Input → Text Extraction → AI Processing → Confidence Scoring → Optional Enrichment → Cached Result
```

### Core Components

- **Pipeline**: Main orchestration (`ExtractionPipeline`)
- **Services**: Domain, AI, Text, Cache services
- **Models**: Pydantic models for type safety
- **Plugins**: Extensible plugin system
- **Configuration**: Centralized configuration management

## Modules

### Core Services

#### `models.py`
Pydantic models for type safety and validation:
- `ExtractionOptions`: Configuration for extraction
- `ExtractionResult`: Complete extraction result
- `DocumentType`: Supported document types enum
- `ValidationResult`: Validation errors and warnings
- `ConfidenceScore`: Confidence scoring results

#### `pipeline.py`
Main orchestration pipeline:
- `ExtractionPipeline`: Coordinates all services
- Handles caching, AI processing, and plugin execution
- Provides unified interface for extraction

#### `domain_service.py`
Document domain management:
- `DomainService`: Manages document types and field schemas
- Domain normalization and field validation
- Schema management for different document types

#### `ai_service.py`
AI-powered extraction:
- `AIExtractionService`: Handles AI provider integration
- Supports Ollama, OpenAI, and Anthropic
- Prompt building and response parsing

#### `text_service.py`
Text extraction from PDFs:
- `TextExtractionService`: PDF text extraction
- Parallel page processing
- OCR integration for scanned documents

#### `cache_service.py`
Result caching:
- `CacheService`: Memory and disk caching
- Cache key generation and TTL management
- Performance optimization

#### `confidence.py`
Confidence scoring:
- `ConfidenceScorer`: Computes confidence scores
- Domain-specific scoring rules
- Field-level confidence assessment

### Configuration

#### `config.py`
Centralized configuration:
- `ExtractionConfig`: Pydantic-based configuration
- Environment variable support
- Service-specific configuration methods

### Plugin System

#### `plugins/base.py`
Plugin infrastructure:
- `ExtractorPlugin`: Abstract base class
- `PluginRegistry`: Plugin management
- Plugin discovery and execution

#### `plugins/table_extractor.py`
Table extraction plugin:
- `TableExtractorPlugin`: Extracts tables from PDFs
- Placeholder for pdfplumber integration
- Pattern-based table detection

#### `plugins/formula_extractor.py`
Formula extraction plugin:
- `FormulaExtractorPlugin`: Extracts mathematical formulas
- LaTeX and text-based formula detection
- Formula complexity classification

## Usage Examples

### Basic Extraction

```python
from extraction.pipeline import ExtractionPipeline
from extraction.models import ExtractionOptions

# Initialize pipeline (done automatically in app.py)
pipeline = ExtractionPipeline(...)

# Configure extraction
options = ExtractionOptions(
    domain_override="invoice",
    selected_fields=["invoice_number", "total_amount"],
    enrich=True
)

# Extract from PDF
result = pipeline.extract("document.pdf", options)

# Access results
print(f"Document type: {result.type}")
print(f"Confidence: {result.confidence.overall}")
print(f"Fields: {result.mapped_fields}")
```

### Custom Plugin Development

```python
from extraction.plugins.base import ExtractorPlugin
from extraction.models import ExtractionOptions, PluginMetadata

class CustomExtractorPlugin(ExtractorPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_extractor",
            version="1.0.0",
            description="Custom extraction plugin",
            supported_types=["invoice", "contract"],
            priority=50
        )
    
    def can_handle(self, document_type: str, options: ExtractionOptions) -> bool:
        return document_type in ["invoice", "contract"]
    
    def extract(self, text: str, pages_text: list[str], options: ExtractionOptions) -> dict:
        # Custom extraction logic
        return {
            "custom_field": "extracted_value",
            "plugin": "custom_extractor"
        }
```

### Configuration

```python
from extraction.config import config

# Access configuration
ai_config = config.get_ai_config()
cache_config = config.get_cache_config()

# Environment variables (with EXTRACTION_ prefix)
# EXTRACTION_CACHE_TTL_SECONDS=86400
# EXTRACTION_MAX_TEXT_LENGTH=20000
# EXTRACTION_AI_TIMEOUT=180
```

## Document Types

The system supports various document types with specific field schemas:

### Invoice
- `invoice_number`, `total_amount`, `vendor_name`, `customer_name`
- `subtotal_amount`, `tax_amount`, `due_date`, `payment_terms`

### Research Paper
- `title`, `authors`, `abstract`, `methodology`, `results`
- `conclusions`, `doi`, `citations`, `references`

### Healthcare
- `patient_id`, `mrn`, `chief_complaint`, `history`
- `assessment`, `plan`, `medications`, `allergies`

### Contract
- `party_a`, `party_b`, `effective_date`, `term`
- `governing_law`, `jurisdiction`, `payment_terms`

### Financial
- `statement_type`, `period`, `revenue`, `net_income`
- `total_assets`, `total_liabilities`, `ratios`

## API Integration

The extraction module integrates with Flask routes in `app.py`:

```python
# Automatic service initialization
_extraction_pipeline = ExtractionPipeline(...)

# Route integration
@app.post("/api/extract")
def api_extract():
    options = ExtractionOptions(...)
    result = _extraction_pipeline.extract(temp_path, options)
    return jsonify(result.to_dict())
```

## Testing

The module includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_domain_service.py

# Run with coverage
pytest --cov=extraction tests/
```

### Test Structure
- `test_models.py`: Pydantic model tests
- `test_domain_service.py`: Domain service tests
- `test_ai_service.py`: AI service tests (mocked)
- `test_pipeline.py`: Pipeline orchestration tests
- `test_integration.py`: End-to-end integration tests
- `conftest.py`: Shared fixtures and configuration

## Performance Considerations

### Caching
- Memory cache for frequently accessed results
- Disk cache for persistence across restarts
- Cache invalidation based on pipeline version

### Parallel Processing
- Parallel page text extraction
- Configurable worker thread count
- Async processing support

### Text Optimization
- Smart text chunking for large documents
- Token limit management
- OCR fallback for scanned documents

## Error Handling

The system includes comprehensive error handling:

- Graceful degradation when services are unavailable
- Fallback to legacy extraction logic
- Detailed error reporting and logging
- Timeout handling for AI services

## Future Enhancements

### Planned Features
- Additional document type support
- Enhanced table extraction with pdfplumber
- Formula extraction improvements
- Custom field extraction via plugins
- Batch processing optimization

### Plugin Development
- Dynamic plugin loading
- Plugin marketplace
- Custom extraction rules
- Domain-specific plugins

## Contributing

When contributing to the extraction module:

1. Follow the existing architecture patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility
5. Use type hints and Pydantic models

## Dependencies

Core dependencies:
- `pydantic>=2.0.0`: Data validation and serialization
- `PyPDF2`: PDF text extraction
- `requests`: HTTP client for AI services
- `pytest>=7.0.0`: Testing framework

Optional dependencies:
- `pdfplumber`: Enhanced table extraction
- `pytesseract`: OCR capabilities
