# Perk PDF

A web-based PDF toolbox that provides various PDF manipulation and conversion tools.

## Local Development Setup

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the backend directory:
   ```
   GMAIL_APP_PASSWORD=your_gmail_app_password
   ```

4. Run the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Create a `.env` file in the frontend directory:
   ```
   VITE_API_BASE=http://localhost:5000
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

## Deployment

### Backend (Render)
1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Add environment variables in Render dashboard:
   - `GMAIL_APP_PASSWORD`: Your Gmail App Password

### Frontend (Vercel)
1. Push your code to GitHub
2. Create a new project on Vercel
3. Connect your GitHub repository
4. Add environment variables in Vercel dashboard:
   - `VITE_API_BASE`: Your Render backend URL

## Security Notes
- Never commit `.env` files to version control
- Keep your Gmail App Password secure
- Use environment variables for all sensitive information
- Regularly rotate your App Password for security

## Features
- PDF Merge
- PDF Split
- PDF Compression
- PDF to Word
- PDF to Images
- Images to PDF
- PDF to Text
- PDF to Excel
- Delete Pages
- Reorder Pages
- Contact Form 

## Extraction

The extraction system has been completely refactored into a modular, scalable architecture with clear separation of concerns.

### Architecture Overview

The extraction system uses a pipeline-based architecture with the following components:

- **Pipeline**: Main orchestration (`ExtractionPipeline`) that coordinates all services
- **Services**: Domain, AI, Text, Cache services for specific functionality
- **Models**: Pydantic models for type safety and validation
- **Plugins**: Extensible plugin system for additional extraction capabilities
- **Configuration**: Centralized configuration management

### Core Services

- **Domain Service**: Manages document types and field schemas
- **AI Service**: Handles AI provider integration (Ollama, OpenAI, Anthropic)
- **Text Service**: PDF text extraction with OCR fallback
- **Cache Service**: Memory and disk caching for performance
- **Confidence Scorer**: Computes confidence scores for extracted fields

### Plugin System

The system includes a plugin architecture for extensibility:

- **Table Extractor**: Extracts tables from PDFs (placeholder for pdfplumber integration)
- **Formula Extractor**: Extracts mathematical formulas and equations
- **Custom Plugins**: Easy to add new extraction capabilities

### API Endpoints

- **Sync `/api/extract`**: Uses the new pipeline architecture with caching and plugin support
- **Async `/api/extract/async`**: Supports OCR via `use_ocr=1` flag
- **Batch `/api/extract/batch`**: Batch processing with pipeline integration
- **PDF Report `/api/extract/pdf`**: Generates PDF reports with extracted data

### Configuration

The system supports extensive configuration via environment variables:

```bash
# Cache settings
EXTRACTION_CACHE_TTL_SECONDS=86400
EXTRACTION_CACHE_ENABLED=true

# AI settings
EXTRACTION_AI_SERVICE=ollama
EXTRACTION_AI_TIMEOUT=180
EXTRACTION_MAX_TEXT_LENGTH=20000

# Plugin settings
EXTRACTION_PLUGINS_ENABLED=true
```

### Document Types

Supports various document types with specific field schemas:

- **Invoice**: `invoice_number`, `total_amount`, `vendor_name`, etc.
- **Research**: `title`, `authors`, `abstract`, `methodology`, etc.
- **Healthcare**: `patient_id`, `mrn`, `chief_complaint`, etc.
- **Contract**: `party_a`, `party_b`, `effective_date`, etc.
- **Financial**: `statement_type`, `revenue`, `net_income`, etc.

### Usage Example

```python
from extraction.pipeline import ExtractionPipeline
from extraction.models import ExtractionOptions

# Configure extraction
options = ExtractionOptions(
    domain_override="invoice",
    selected_fields=["invoice_number", "total_amount"],
    enrich=True
)

# Extract from PDF
result = pipeline.extract("document.pdf", options)
```

### Testing

The extraction module includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run specific modules
pytest tests/test_models.py
pytest tests/test_domain_service.py
```

For detailed documentation, see [extraction/README.md](extraction/README.md).

## License

Licensed under AGPL-3.0-only. See the `LICENSE` file for details.

Copyright (c) 2025 Arash Sheikhlar 