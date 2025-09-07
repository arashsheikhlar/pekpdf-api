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

## License

Licensed under AGPL-3.0-only. See the `LICENSE` file for details.

Copyright (c) 2025 Arash Sheikhlar 