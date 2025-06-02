# 1. Base image: slim Python 3.11
FROM python:3.11-slim

# 2. Install system packages: Tesseract, Ghostscript, Poppler, pngquant
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         tesseract-ocr \
         libtesseract-dev \
         ghostscript \
         poppler-utils \
         pngquant \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements and install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . .

# 5. Expose port 8000 (Flask default)
EXPOSE 8000

# 6. Start the Flask app
CMD ["python", "app.py"]
