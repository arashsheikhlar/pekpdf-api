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

Current extraction is AI-only and schema-driven. The classic heuristic pipeline and specialized table/formula extractors were removed for simplicity and maintainability. OCR remains available in the async extraction endpoint via `OCRService`.

- Sync `/api/extract`: reads native PDF text, accepts `domain_override` and optional `custom_instructions`, and calls the AI extractor. Confidence is computed with `ConfidenceScorer`.
- Async `/api/extract/async`: supports an optional `use_ocr=1` flag; when enabled, uses a hybrid native-text + OCR flow before AI extraction. Confidence is computed with `ConfidenceScorer`.

Notes:
- `extraction/confidence.py` exports `ConfidenceScorer`.
- `extraction/specialized.py` was removed (no tables/formulas wiring at this time).

## License

Licensed under AGPL-3.0-only. See the `LICENSE` file for details.

Copyright (c) 2025 Arash Sheikhlar 