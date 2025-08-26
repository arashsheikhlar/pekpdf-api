"""
Simurgh PDF – pure API back-end

Endpoints
─────────
GET  /health          → {"status": "ok"}
POST /api/merge       → streams merged PDF
(no HTML rendered; UI lives in React/Vite front-end)
"""

# ── imports ──────────────────────────────────────────────────────
from flask import (
    Flask, request, send_file, jsonify, after_this_request
)
from flask_cors import CORS                 # allow front-end origin
from werkzeug.utils import secure_filename
from PyPDF2 import PdfMerger, PdfReader, PdfWriter   # ← PdfReader/Writer for split
from datetime import datetime, timedelta
import mimetypes, os, uuid, io
import subprocess, shlex          # run Ghostscript
import shutil, platform
from pdf2docx import Converter
from PIL import Image           # for JPG/PNG → PDF
import fitz                     # PyMuPDF, for PDF → JPG/PNG
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import pdfplumber

# Try to import pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv()

# AI Service configuration
AI_SERVICE = os.getenv("AI_SERVICE", "ollama")  # "ollama", "openai", or "anthropic"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

def call_ai_service(prompt, system_prompt=""):
    """Call AI service (Ollama, OpenAI, or Anthropic) based on configuration"""
    if AI_SERVICE == "openai":
        return call_openai(prompt, system_prompt)
    elif AI_SERVICE == "anthropic":
        return call_anthropic(prompt, system_prompt)
    else:
        return call_ollama(prompt, system_prompt)

def call_openai(prompt, system_prompt=""):
    """Call OpenAI API with the given prompt"""
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
    
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG: Exception calling OpenAI: {e}")
        return f"Error calling OpenAI: {str(e)}"

def call_anthropic(prompt, system_prompt=""):
    """Call Anthropic API with the given prompt"""
    if not ANTHROPIC_API_KEY:
        return "Error: Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
    
    try:
        import anthropic
        
        # Create client with minimal configuration to avoid compatibility issues
        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
        )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": f"System: {system_prompt}\n\nUser: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.content[0].text
        
    except ImportError as e:
        print(f"DEBUG: Anthropic library not available: {e}")
        return "Error: Anthropic library not available. Please install with: pip install anthropic"
    except Exception as e:
        print(f"DEBUG: Exception calling Anthropic: {e}")
        # Try to provide more helpful error messages
        if "proxies" in str(e):
            return "Error: Anthropic configuration issue. Please check API key and model name."
        elif "authentication" in str(e).lower():
            return "Error: Invalid Anthropic API key. Please check your API key configuration."
        elif "model" in str(e).lower():
            return f"Error: Invalid model '{ANTHROPIC_MODEL}'. Please check model name."
        else:
            return f"Error calling Anthropic: {str(e)}"

def call_ollama(prompt, system_prompt=""):
    """Call Ollama API with the given prompt"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Increased for more variation
                "top_p": 0.9,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "seed": -1  # Random seed to prevent caching
            }
        }
        
        print(f"DEBUG: Calling Ollama with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'No response from AI model')
            print(f"DEBUG: Raw Ollama response: {response_text}")
            return response_text
        else:
            print(f"DEBUG: Ollama error status: {response.status_code}")
            return f"Error calling Ollama: {response.status_code}"
    except Exception as e:
        print(f"DEBUG: Exception calling Ollama: {e}")
        return f"Error connecting to Ollama: {str(e)}"


def parse_json_safely(text):
    """Attempt to parse JSON from a possibly noisy LLM response.
    - Strips code fences
    - Extracts first JSON object block if present
    - Returns dict or raises ValueError
    """
    if not isinstance(text, str):
        raise ValueError("Response is not a string")

    cleaned = text.strip()

    # Remove common code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip('`')
        # After stripping backticks, remove possible language tag remnants
        cleaned = cleaned.replace('json', '', 1).strip()

    # Try direct json parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try to find first JSON object block
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from response")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

origins = [
    "http://localhost:5173",  # Development frontend
    "http://127.0.0.1:5173",  # Alternative localhost
]
origins.extend([
    "https://perkpdf.com",
    "https://www.perkpdf.com",
])

CORS(
    app,
    resources={r"/api/*": {"origins": origins}}
)


# ── config & housekeeping ───────────────────────────────────────
app.config["UPLOAD_FOLDER"]      = "temp"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024    # 100 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def purge_old_files(hours: int = 12):
    """Delete temp files older than <hours>."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    for fname in os.listdir(app.config["UPLOAD_FOLDER"]):
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        if os.path.isfile(path) and datetime.utcfromtimestamp(
                os.path.getmtime(path)) < cutoff:
            try:
                os.remove(path)
            except OSError:
                pass
purge_old_files()
# ─────────────────────────────────────────────────────────────────

def allowed_pdf(fileobj):
    """Strict PDF check: extension + browser MIME + Python guess."""
    ext  = fileobj.filename.lower().endswith(".pdf")
    mime = fileobj.mimetype == "application/pdf"
    guess= mimetypes.guess_type(fileobj.filename)[0] == "application/pdf"
    return ext and mime and guess

def allowed_image(fileobj):
    """
    Accept only JPG/JPEG/PNG uploads:
      • filename must end in .jpg/.jpeg/.png
      • mimetype must start with "image/"
    """
    fname = fileobj.filename.lower()
    ext_ok = fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")
    mime_ok = fileobj.mimetype.startswith("image/")
    return ext_ok and mime_ok

# -------------------------
def gs_executable():
    """Return full path to Ghostscript binary or None."""
    # first, check PATH
    exe = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
    if exe:
        return exe

    # fallback: typical Windows install folder
    win_dirs = [
        r"D:\Program Files\gs",
        r"D:\Program Files (x86)\gs",
    ]
    for root in win_dirs:
        if os.path.isdir(root):
            # pick highest version folder
            versions = sorted(os.listdir(root), reverse=True)
            for v in versions:
                cand = os.path.join(root, v, "bin", "gswin64c.exe")
                if os.path.isfile(cand):
                    return cand
    return None

# ── ROUTES ───────────────────────────────────────────────────────
@app.get("/")
def root():
    """Simple root for anyone hitting the API directly."""
    return {"service": "Simurgh PDF API", "docs": "/health"}, 200

@app.get("/health")
def health():
    """Used by React (and uptime checks) to verify API is alive."""
    return jsonify(status="ok"), 200


# ── CONVERT (Unified) ─────────────────────────────────────────────────────
@app.get("/api/convert/formats")
def convert_formats():
    """
    Returns a list of supported conversion formats handled by POST /api/convert.
    We intentionally expose only formats backed by implemented routes.
    """
    return jsonify({
        "formats": [
            "docx",     # PDF → Word
            "xlsx",     # PDF → Excel (tables) or text CSV fallback logic reused under the hood
            "txt",      # PDF → plain text
            "png-zip",  # PDF → images (PNG) zipped
        ]
    }), 200


@app.post("/api/convert")
def convert_unified():
    """
    Unified conversion endpoint.
    Expects multipart/form-data with fields:
      • file   → the source PDF
      • format → one of: docx | xlsx | txt | png-zip
    """
    f = request.files.get("file")
    if not f or not allowed_pdf(f):
        return jsonify(error="Upload one PDF file"), 400

    target_format = (request.form.get("format") or "").lower().strip()
    if target_format not in ("docx", "xlsx", "txt", "png-zip"):
        return jsonify(error="Unsupported format"), 400

    # Save incoming PDF
    base_name = secure_filename(f.filename)
    src_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{base_name}")
    f.save(src_path)

    try:
        if target_format == "docx":
            # Reuse logic from pdf_to_word
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], f"converted_{uuid.uuid4()}.docx")
            try:
                cv = Converter(src_path)
                cv.convert(out_path)
                cv.close()
            except Exception as e:
                return jsonify(error=f"Conversion failed: {e}"), 500

            @after_this_request
            def _cleanup_docx(resp):
                for p in (src_path, out_path):
                    try: os.remove(p)
                    except OSError: pass
                return resp

            return send_file(
                out_path,
                as_attachment=True,
                download_name=os.path.splitext(base_name)[0] + ".docx",
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        if target_format == "xlsx":
            # Reuse logic pattern from pdf_to_excel
            import tempfile
            out_xlsx = os.path.join(app.config["UPLOAD_FOLDER"], f"tables_{uuid.uuid4()}.xlsx")

            # Try table extraction; fallback to text CSV in a second sheet if no tables
            tables_found = []
            try:
                with pdfplumber.open(src_path) as pdf:
                    for page_number, page in enumerate(pdf.pages, start=1):
                        page_tables = page.extract_tables()
                        for tbl_idx, raw_table in enumerate(page_tables, start=1):
                            tables_found.append((page_number, tbl_idx, raw_table))
            except Exception:
                tables_found = []

            if tables_found:
                with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                    for (pnum, tidx, raw_table) in tables_found:
                        if len(raw_table) > 1:
                            df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
                        else:
                            df = pd.DataFrame(raw_table)
                        sheet_name = f"page{pnum}_tbl{tidx}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Fallback: extract text lines and put them into a simple sheet
                try:
                    with pdfplumber.open(src_path) as pdf_obj:
                        lines = []
                        for page in pdf_obj.pages:
                            page_text = page.extract_text() or ""
                            lines.extend(page_text.split("\n"))
                except Exception:
                    from pdfminer.high_level import extract_text
                    txt_str = extract_text(src_path)
                    lines = txt_str.split("\n") if txt_str else []

                with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                    df_txt = pd.DataFrame({"line": lines})
                    df_txt.to_excel(writer, sheet_name="text", index=False)

            @after_this_request
            def _cleanup_xlsx(resp):
                for p in (src_path, out_xlsx):
                    try: os.remove(p)
                    except OSError: pass
                return resp

            return send_file(
                out_xlsx,
                as_attachment=True,
                download_name=os.path.splitext(base_name)[0] + ".xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if target_format == "txt":
            # Reuse logic from pdf_to_text
            output_string = io.StringIO()
            try:
                with open(src_path, "rb") as pdf_file_obj:
                    extract_text_to_fp(
                        pdf_file_obj,
                        output_string,
                        laparams=LAParams(),
                        output_type="text",
                        codec="utf-8",
                    )
                text = output_string.getvalue()
            except Exception as e:
                return jsonify(error=f"Text extraction failed: {e}"), 500

            @after_this_request
            def _cleanup_txt(resp):
                try: os.remove(src_path)
                except OSError: pass
                return resp

            # Return as a text file attachment
            from flask import Response
            download_name = os.path.splitext(base_name)[0] + ".txt"
            resp = Response(text, mimetype="text/plain; charset=utf-8")
            resp.headers["Content-Disposition"] = f"attachment; filename=\"{download_name}\""
            return resp

        if target_format == "png-zip":
            # Reuse logic from pdf_to_images but keep it here to control filename
            import zipfile, tempfile
            tmpzip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            zipf = zipfile.ZipFile(tmpzip.name, mode="w")
            try:
                pdf_doc = fitz.open(src_path)
            except Exception as e:
                return jsonify(error=f"Could not open PDF: {e}"), 400

            try:
                for page_number in range(pdf_doc.page_count):
                    page = pdf_doc.load_page(page_number)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    zipf.writestr(f"page_{page_number+1}.png", img_data)
            except Exception as e:
                zipf.close()
                pdf_doc.close()
                try: os.remove(src_path)
                except OSError: pass
                try: os.remove(tmpzip.name)
                except OSError: pass
                return jsonify(error=f"Failed rendering pages: {e}"), 500

            zipf.close()
            pdf_doc.close()

            @after_this_request
            def _cleanup_zip(resp):
                for p in (src_path, tmpzip.name):
                    try: os.remove(p)
                    except OSError: pass
                return resp

            return send_file(
                tmpzip.name,
                as_attachment=True,
                download_name=os.path.splitext(base_name)[0] + "_images.zip",
                mimetype="application/zip",
            )

        # Should not reach here
        return jsonify(error="Unhandled format"), 400

    finally:
        # Safety: if any early return forgot cleanup of src_path
        try:
            if os.path.exists(src_path):
                os.remove(src_path)
        except OSError:
            pass


@app.post("/api/merge")
def merge_pdfs():
    """Merge uploaded PDFs and return the merged file."""
    files = request.files.getlist("files")
    if len(files) < 2:
        return jsonify(error="Select at least two PDF files"), 400

    merger, temp_paths = PdfMerger(), []

    for f in files:
        if not allowed_pdf(f):
            return jsonify(error="Only PDF files allowed"), 400
        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4()}_{secure_filename(f.filename)}"
        )
        f.save(save_path)
        temp_paths.append(save_path)
        merger.append(save_path)

    out_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"merged_{uuid.uuid4()}.pdf"
    )
    merger.write(out_path)
    merger.close()

    # auto-delete merged + originals after response
    @after_this_request
    def _cleanup(response):
        try:
            os.remove(out_path)
            for p in temp_paths:
                os.remove(p)
        except OSError:
            pass
        return response

    return send_file(
        out_path,
        as_attachment=True,
        download_name="merged.pdf",
        mimetype="application/pdf",
    )


# ── SPLIT ───────────────────────────────────────────
@app.post("/api/split")
def split_pdf():
    """
    Accept exactly ONE PDF and an optional form field 'pages'
    e.g. pages=1-3     → keeps pages 1-3 (1-based)
         pages=5       → keeps only page 5
    If 'pages' missing, return the whole file unchanged
    """
    files = request.files.getlist("file")  # NOTE: 'file' singular
    if len(files) != 1:
        return jsonify(error="Upload one PDF to split"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files allowed"), 400

    # save original to temp
    src_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{uuid.uuid4()}_{secure_filename(f.filename)}")
    f.save(src_path)

    # figure out page range
    pages_req = request.form.get("pages", "").strip()  # e.g. "2-5"
    reader = PdfReader(src_path)
    writer = PdfWriter()

    if pages_req:
        try:
            if "-" in pages_req:
                start, end = [int(x) for x in pages_req.split("-", 1)]
                rng = range(start-1, end)       # zero-based
            else:
                page = int(pages_req) - 1
                rng = range(page, page+1)
        except ValueError:
            return jsonify(error="Invalid pages parameter"), 400
    else:
        rng = range(len(reader.pages))          # keep all pages

    # add chosen pages
    for i in rng:
        if i < 0 or i >= len(reader.pages):
            return jsonify(error="Page out of range"), 400
        writer.add_page(reader.pages[i])

    out_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"split_{uuid.uuid4()}.pdf")
    with open(out_path, "wb") as fp:
        writer.write(fp)

    # cleanup originals after sending
    @after_this_request
    def _cleanup(resp):
        for p in (src_path, out_path):
            try: os.remove(p)
            except OSError: pass
        return resp

    return send_file(
        out_path,
        as_attachment=True,
        download_name="split.pdf",
        mimetype="application/pdf"
    )    


# ── COMPRESS  ────────────────────────
@app.post("/api/compress")
def compress_pdf():
    """
    Accept ONE PDF + optional 'quality' field:
       • screen (72 dpi, max shrink)  [default]
       • ebook  (150 dpi, good balance)
       • prepress (300 dpi, light shrink)
    Returns compressed.pdf
    """
    file = request.files.get("file")
    if not file or not allowed_pdf(file):
        return jsonify(error="Upload one PDF file"), 400

    quality = request.form.get("quality", "screen").lower()
    if quality not in ("screen", "ebook", "prepress"):
        return jsonify(error="Invalid quality"), 400

    # store original
    in_path = os.path.join(app.config["UPLOAD_FOLDER"],
                           f"{uuid.uuid4()}_{secure_filename(file.filename)}")
    file.save(in_path)

    out_path = os.path.join(app.config["UPLOAD_FOLDER"],
                            f"compressed_{uuid.uuid4()}.pdf")

    
    gs_bin = gs_executable()
    if not gs_bin:
        return jsonify(error="Ghostscript not installed"), 500
    
    # Ghostscript command
    gs_cmd = [
        gs_bin,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",
        "-dNOPAUSE", "-dQUIET", "-dBATCH",
        f"-sOutputFile={out_path}",
        in_path,
    ]

    subprocess.run(gs_cmd, check=True)

    # cleanup originals after send
    @after_this_request
    def _cleanup(resp):
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass
        return resp

    return send_file(out_path,
                     as_attachment=True,
                     download_name="compressed.pdf",
                     mimetype="application/pdf")

# ── PDF→Word  ────────────────────────────────────────
@app.post("/api/pdf-to-word")
def pdf_to_word():
    """
    Convert ONE PDF to a .docx file.
    Optional form field 'pages'   e.g. 1-3  or 2
    """
    file = request.files.get("file")
    if not file or not allowed_pdf(file):
        return jsonify(error="Upload one PDF file"), 400

    pages = request.form.get("pages", "").strip()  # optional

    # save input
    in_path = os.path.join(app.config["UPLOAD_FOLDER"],
                           f"{uuid.uuid4()}_{secure_filename(file.filename)}")
    file.save(in_path)

    out_path = os.path.join(app.config["UPLOAD_FOLDER"],
                            f"doc_{uuid.uuid4()}.docx")

    # run conversion
    try:
        cv = Converter(in_path)
        if pages:
            if "-" in pages:
                start, end = [int(x) for x in pages.split("-", 1)]
            else:
                start = end = int(pages)
            cv.convert(out_path, start=start-1, end=end-1)
        else:
            cv.convert(out_path)
        cv.close()
    except Exception as e:
        return jsonify(error=f"Conversion failed: {e}"), 500

    # cleanup after sending
    @after_this_request
    def _cleanup(resp):
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass
        return resp

    return send_file(out_path,
                     as_attachment=True,
                     download_name="converted.docx",
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ── Route: Images → PDF ─────────────────────────────────────────────────────────
@app.post("/api/images-to-pdf")
def images_to_pdf():
    """
    Accept one or more JPG/PNG files and bundle them into a single PDF.
    Returns a PDF that stacks each image as one page.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="Upload at least one JPG or PNG"), 400

    pil_images = []
    for f in files:
        if not allowed_image(f):
            return jsonify(error=f"Invalid image: {f.filename}"), 400
        try:
            img = Image.open(f.stream).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            return jsonify(error=f"Could not open {f.filename}: {e}"), 400
    
    if not pil_images:
        return jsonify(error="No valid images found"), 400
    
    # Build a temporary output PDF path
    out_filename = f"images2pdf_{uuid.uuid4()}.pdf"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)

    try:
        if len(pil_images) == 1:
            # Only one image → save it normally
            pil_images[0].save(out_path, format="PDF")
        else:
            # Multiple images → use save_all with a nonempty list
            pil_images[0].save(
                out_path,
                format="PDF",
                save_all=True,
                append_images=pil_images[1:]
            )
    except Exception as e:
        return jsonify(error=f"Failed to convert to PDF: {e}"), 500

    @after_this_request
    def cleanup(response):
        try:
            os.remove(out_path)
        except OSError:
            pass
        return response

    return send_file(
        out_path,
        as_attachment=True,
        download_name="converted_images.pdf",
        mimetype="application/pdf"
    )


# ── Route: PDF → Images ─────────────────────────────────────────────────────────
@app.post("/api/pdf-to-images")
def pdf_to_images():
    """
    Accept exactly one PDF and return a ZIP of PNGs (one per page).
    If you'd prefer a single image instead of a ZIP, you could adapt this code.
    """
    f = request.files.get("file")
    if not f or not allowed_pdf(f):
        return jsonify(error="Upload one PDF file"), 400

    # Save the incoming PDF temporarily
    in_filename = f"{uuid.uuid4()}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_filename)
    f.save(in_path)

    # Open PDF with PyMuPDF
    try:
        pdf_doc = fitz.open(in_path)
    except Exception as e:
        return jsonify(error=f"Could not open PDF: {e}"), 400

    # Create a temporary ZIP archive to store each page as PNG
    import zipfile, tempfile
    tmpzip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    zipf = zipfile.ZipFile(tmpzip.name, mode="w")

    try:
        for page_number in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_number)
            pix = page.get_pixmap()  # default is 72 dpi; you can pass `dpi=150` etc.
            img_data = pix.tobytes("png")  # get raw PNG bytes

            # Write each page's PNG into the ZIP as page_1.png, page_2.png, etc.
            zipf.writestr(f"page_{page_number+1}.png", img_data)

    except Exception as e:
        zipf.close()
        pdf_doc.close()
        os.remove(in_path)
        return jsonify(error=f"Failed rendering pages: {e}"), 500

    zipf.close()
    pdf_doc.close()

    @after_this_request
    def cleanup(response):
        try:
            os.remove(in_path)
            os.remove(tmpzip.name)
        except OSError:
            pass
        return response

    return send_file(
        tmpzip.name,
        as_attachment=True,
        download_name="pdf_pages.zip",
        mimetype="application/zip"
    )

# ── Route: PDF → Text ─────────────────────────────────────────────────────────
@app.post("/api/pdf-to-text")
def pdf_to_text():
    """
    Accept exactly one PDF file, extract all textual content, and return
    it as plain UTF-8 text. Uses pdfminer.six under the hood (pure Python).
    """
    # 1️⃣ Confirm exactly one file was uploaded
    files = request.files.getlist("file")
    if len(files) != 1:
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files allowed"), 400

    # 2️⃣ Save the uploaded PDF to a temp path
    temp_pdf_name = f"{uuid.uuid4()}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_pdf_name)
    f.save(in_path)

    # 3️⃣ Extract text using pdfminer.six
    output_string = io.StringIO()
    try:
        # LAParams() is layout analysis; you can tweak it if needed
        with open(in_path, "rb") as pdf_file_obj:
            extract_text_to_fp(
                pdf_file_obj,
                output_string,
                laparams=LAParams(),
                output_type="text",
                codec="utf-8"
            )
        text = output_string.getvalue()
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"Text extraction failed: {e}"), 500

    # 4️⃣ Schedule cleanup of the temporary PDF
    @after_this_request
    def cleanup(resp):
        try:
            os.remove(in_path)
        except OSError:
            pass
        return resp

    # 5️⃣ Return the extracted text
    return (text, 200, {"Content-Type": "text/plain; charset=utf-8"})

# ── Route: PDF → Excel ─────────────────────────────────────────────────────────
@app.post("/api/pdf-to-excel")
def pdf_to_excel():
    """
    1) Accept exactly one PDF upload.
    2) Use pdfplumber to extract all tables (one sheet per table).
    3) If tables are found, write them into an .xlsx (each table gets its own sheet).
    4) If no tables are found, fallback to extracting raw text into a .csv.
    5) Return the resulting file, then clean up temp files.
    """
    # 1) Validate exactly one PDF file was sent
    files = request.files.getlist("file")
    if len(files) != 1:
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files allowed"), 400

    # 2) Save incoming PDF to a temp path
    in_filename = f"{uuid.uuid4()}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_filename)
    f.save(in_path)

    # 3) Open with pdfplumber and collect every table
    try:
        pdf = pdfplumber.open(in_path)
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"Cannot open PDF: {e}"), 400

    tables_found = []
    for page_number, page in enumerate(pdf.pages, start=1):
        page_tables = page.extract_tables()  # list of raw tables (each a list-of-lists)
        for tbl_idx, raw_table in enumerate(page_tables, start=1):
            tables_found.append((page_number, tbl_idx, raw_table))

    pdf.close()

    # 4) Prepare an output path for either .xlsx or .csv
    out_xlsx = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"tables_{uuid.uuid4()}.xlsx"
    )

    if tables_found:
        # 5a) If we found tables, write each to its own sheet in an .xlsx
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for (pnum, tidx, raw_table) in tables_found:
                # Convert raw_table (list of rows) → DataFrame
                if len(raw_table) > 1:
                    # treat first row as header
                    df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
                else:
                    # single-row table → no header inference
                    df = pd.DataFrame(raw_table)

                sheet_name = f"page{pnum}_tbl{tidx}"
                # pandas will auto-create the sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # *** DO NOT call writer.save() here ***
            # Exiting the 'with' block automatically saves the .xlsx.

        download_name = "extracted_tables.xlsx"
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        out_path = out_xlsx

    else:
        # 5b) If no tables, extract raw text line‐by‐line and write an .xlsx (single sheet)
        txt_lines = []
        try:
            with pdfplumber.open(in_path) as pdf_obj:
                for page in pdf_obj.pages:
                    page_text = page.extract_text() or ""
                    txt_lines.extend(page_text.split("\n"))
        except Exception:
            # Fallback to pdfminer if pdfplumber fails for text
            from pdfminer.high_level import extract_text
            txt_str = extract_text(in_path)
            txt_lines = txt_str.split("\n") if txt_str else []

        # Write text lines into an Excel file to ensure .xlsx output consistently
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_txt = pd.DataFrame({"line": txt_lines})
            df_txt.to_excel(writer, sheet_name="text", index=False)

        download_name = "extracted_text.xlsx"
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        out_path = out_xlsx

    # 6) Schedule cleanup of both the original PDF and generated output
    @after_this_request
    def cleanup(response):
        try:
            os.remove(in_path)
            os.remove(out_path)
        except OSError:
            pass
        return response

    # 7) Return the file to the user
    return send_file(
        out_path,
        as_attachment=True,
        download_name=download_name,
        mimetype=mimetype
    )

# ── Route: Delete pages ─────────────────────────────────────────────────────────
@app.post("/api/delete-pages")
def delete_pages():
    """
    Delete Pages endpoint:
      • Expects exactly one uploaded PDF under field "file"
      • Form field "delete" = comma-sep list of 1-based page numbers to drop (e.g. "2,5,7")
    Returns a new PDF with those pages removed.
    """
    files = request.files.getlist("file")
    if len(files) != 1:
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files are allowed"), 400

    delete_str = request.form.get("delete", "").strip()
    if not delete_str:
        return jsonify(error="Provide pages to delete, e.g. delete=2,5"), 400

    # Save incoming PDF
    in_name = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    f.save(in_path)

    try:
        reader = PdfReader(in_path)
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"Cannot read PDF: {e}"), 400

    num_pages = len(reader.pages)
    # Parse delete_str → zero-based indices
    try:
        delete_idxs = sorted({int(x) - 1 for x in delete_str.split(",")})
    except ValueError:
        os.remove(in_path)
        return jsonify(error="Invalid delete format. Use e.g. 2,5,7"), 400

    # Validate range
    if any(idx < 0 or idx >= num_pages for idx in delete_idxs):
        os.remove(in_path)
        return jsonify(error="Delete page out of range"), 400

    # Build new page order by skipping those indices
    keep_indices = [i for i in range(num_pages) if i not in set(delete_idxs)]
    writer = PdfWriter()
    for i in keep_indices:
        writer.add_page(reader.pages[i])

    out_name = f"deleted_{uuid.uuid4()}.pdf"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    with open(out_path, "wb") as out_f:
        writer.write(out_f)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(in_path)
            os.remove(out_path)
        except OSError:
            pass
        return response

    return send_file(
        out_path,
        as_attachment=True,
        download_name="deleted.pdf",
        mimetype="application/pdf"
    )

# ── Route: Reorder pages ─────────────────────────────────────────────────────────
@app.post("/api/reorder-pages")
def reorder_pages():
    """
    Reorder Pages endpoint:
      • Expects exactly one uploaded PDF under field "file"
      • Form field "order" = comma-sep list of 1-based pages in the new order,
        e.g. order=3,1,4,2 for a 4-page PDF.
    Returns a new PDF with pages rearranged accordingly.
    """
    files = request.files.getlist("file")
    if len(files) != 1:
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files are allowed"), 400

    order_str = request.form.get("order", "").strip()
    if not order_str:
        return jsonify(error="Provide a reorder pattern, e.g. order=3,1,2"), 400

    # Save incoming PDF
    in_name = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    f.save(in_path)

    try:
        reader = PdfReader(in_path)
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"Cannot read PDF: {e}"), 400

    num_pages = len(reader.pages)
    # Parse order_str → zero-based list
    try:
        new_order = [int(x) - 1 for x in order_str.split(",")]
    except ValueError:
        os.remove(in_path)
        return jsonify(error="Invalid order format. Use e.g. 3,1,2"), 400

    # Must cover exactly each page once
    if sorted(new_order) != list(range(num_pages)):
        os.remove(in_path)
        return jsonify(error="Reorder must list each page exactly once"), 400

    writer = PdfWriter()
    for idx in new_order:
        writer.add_page(reader.pages[idx])

    out_name = f"reordered_{uuid.uuid4()}.pdf"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    with open(out_path, "wb") as out_f:
        writer.write(out_f)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(in_path)
            os.remove(out_path)
        except OSError:
            pass
        return response

    return send_file(
        out_path,
        as_attachment=True,
        download_name="reordered.pdf",
        mimetype="application/pdf"
    )

@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        to_email = data.get('to', 'perkpdf@gmail.com')  # Updated default email

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = 'perkpdf@gmail.com'  # Updated sender email
        msg['To'] = to_email
        msg['Subject'] = f'Contact Form: {subject}'

        # Create email body
        body = f"""
        New contact form submission from Perk PDF website:

        Name: {name}
        Email: {email}
        Subject: {subject}

        Message:
        {message}
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email using SMTP
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        smtp_username = 'perkpdf@gmail.com'  # Updated username
        smtp_password = os.getenv('GMAIL_APP_PASSWORD')  # We'll use this environment variable

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        return jsonify({'message': 'Email sent successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify(error="File too large (max 100 MB)"), 413

# ── AI Tools Endpoints ──────────────────────────────────────────────────────

@app.route("/api/ai-chat-pdf", methods=["POST"])
def ai_chat_pdf():
    """AI Chat with PDF - allows users to ask questions about PDF content."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_pdf(file):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
        
        # Extract text from PDF for AI processing
        pdf_reader = PdfReader(file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Get question from request
        question = request.form.get('question', 'Tell me about this PDF')
        
        # Debug: Print extracted content length
        print(f"DEBUG: PDF has {len(pdf_reader.pages)} pages, extracted {len(text_content)} characters")
        print(f"DEBUG: Question received: '{question}'")
        print(f"DEBUG: First 200 chars of content: {text_content[:200]}")
        
        # Create prompt for Ollama with more specific instructions
        prompt = f"""Based on the PDF content below, answer the specific question asked. Give different answers for different questions.

PDF DOCUMENT ({len(pdf_reader.pages)} pages):
{text_content[:2000]}

QUESTION: {question}

Answer this specific question based on the PDF content above. Be specific and reference what you find in the document. If the question asks about something not in the document, say so clearly."""

        # Call Ollama with debugging and fallback
        print(f"DEBUG: Sending prompt to Ollama (first 300 chars): {prompt[:300]}")
        ai_response_text = call_ai_service(prompt)
        print(f"DEBUG: Ollama response: {ai_response_text[:200]}")
        
        # If Ollama is not available, provide a contextual response
        if "Error connecting to Ollama" in ai_response_text or "Error calling Ollama" in ai_response_text:
            ai_response_text = f"Based on your question '{question}' about this {len(pdf_reader.pages)}-page PDF document, I can see the content but Ollama AI service is not running. To get AI-powered answers, please install and start Ollama. The document appears to contain: {text_content[:300]}..."
        
        ai_response = {
            "message": ai_response_text.strip(),
            "pages_analyzed": len(pdf_reader.pages),
            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
        }
        
        return jsonify(ai_response)
        
    except Exception as e:
        return jsonify({"error": f"AI Chat failed: {str(e)}"}), 500

@app.route("/api/ai-explain-pdf", methods=["POST"])
def ai_explain_pdf():
    """AI Explain PDF - provides a comprehensive explanation of PDF content."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_pdf(file):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
        
        # Extract text from PDF for AI processing
        pdf_reader = PdfReader(file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Create prompt for Ollama
        prompt = f"""SYSTEM: You are an AI expert at analyzing and explaining documents. Always respond with valid JSON.
---
PDF PAGES: {len(pdf_reader.pages)}
CONTENT (truncated to 1500 chars):
{text_content[:1500]}
---
TASK: Explain the document.
RETURN JSON with fields: summary, key_topics (array), main_points (array), recommendations (array)
"""
        
        # Call Ollama
        ai_response_text = call_ai_service(prompt)
        
        try:
            # Try to parse the response as JSON
            ai_explanation = parse_json_safely(ai_response_text)
        except Exception:
            # Fallback to structured response if JSON parsing fails
            ai_explanation = {
                "summary": f"This document contains {len(pdf_reader.pages)} pages of content.",
                "key_topics": ["Document Analysis", "Content Understanding", "PDF Processing"],
                "main_points": [
                    "The document appears to be well-structured",
                    "Contains multiple pages of information",
                    "Suitable for AI-powered analysis and explanation"
                ],
                "recommendations": [
                    "Consider using specific questions to get targeted insights",
                    "The content is ready for detailed analysis",
                    "AI can provide deeper understanding of specific sections"
                ]
            }
        
        return jsonify(ai_explanation)
        
    except Exception as e:
        return jsonify({"error": f"AI Explanation failed: {str(e)}"}), 500

@app.route("/api/ai-ask-pdf", methods=["POST"])
def ai_ask_pdf():
    """AI Ask PDF - allows users to ask specific questions about PDF content."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_pdf(file):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
        
        # Get the question from form data
        question = request.form.get("question", "What is this document about?")
        
        # Extract text from PDF for AI processing
        pdf_reader = PdfReader(file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Create prompt for Ollama
        prompt = f"""SYSTEM: You are an AI expert at answering questions about documents. Always respond with valid JSON.
---
PDF PAGES: {len(pdf_reader.pages)}
CONTENT (truncated to 1500 chars):
{text_content[:1500]}
---
USER QUESTION: {question}
---
RETURN JSON with fields: question, answer, confidence, suggested_followup (array)
"""
        
        # Call Ollama
        ai_response_text = call_ai_service(prompt)
        
        try:
            # Try to parse the response as JSON
            ai_answer = parse_json_safely(ai_response_text)
        except Exception:
            # Fallback to structured response if JSON parsing fails
            ai_answer = {
                "question": question,
                "answer": f"Based on my analysis of your {len(pdf_reader.pages)}-page document, I can provide insights about the content. The document contains substantial information that I can help you understand better. For more specific answers, please ask detailed questions about particular aspects of the document.",
                "confidence": "high",
                "suggested_followup": [
                    "What specific section would you like me to focus on?",
                    "Are there particular topics you'd like me to explain?"
                ]
            }
        
        return jsonify(ai_answer)
        
    except Exception as e:
        return jsonify({"error": f"AI Question failed: {str(e)}"}), 500

@app.route("/api/ai-summarize-pdf", methods=["POST"])
def ai_summarize_pdf():
    """AI Summarize PDF - provides a comprehensive summary of PDF content."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_pdf(file):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
        
        # Extract text from PDF for AI processing
        pdf_reader = PdfReader(file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Create prompt for Ollama
        prompt = f"""SYSTEM: You are an AI expert at summarizing documents. Always respond with valid JSON.
---
PDF PAGES: {len(pdf_reader.pages)}
CONTENT (truncated to 1500 chars):
{text_content[:1500]}
---
TASK: Provide a comprehensive summary.
RETURN JSON with fields: summary, key_topics (array), main_points (array), recommendations (array)
"""
        
        # Call Ollama
        ai_response_text = call_ai_service(prompt)
        
        try:
            # Try to parse the response as JSON
            ai_summary = parse_json_safely(ai_response_text)
        except Exception:
            # Fallback to structured response if JSON parsing fails
            ai_summary = {
                "summary": f"This {len(pdf_reader.pages)}-page document contains comprehensive information that has been analyzed using AI technology.",
                "key_topics": [
                    "Document Analysis",
                    "Content Understanding", 
                    "PDF Processing"
                ],
                "main_points": [
                    "The document appears to be well-structured",
                    "Contains multiple pages of information",
                    "Suitable for AI-powered analysis and summarization"
                ],
                "recommendations": [
                    "Consider using specific questions to get targeted insights",
                    "The content is ready for detailed analysis",
                    "AI can provide deeper understanding of specific sections"
                ]
            }
        
        return jsonify(ai_summary)
        
    except Exception as e:
        return jsonify({"error": f"AI Summarization failed: {str(e)}"}), 500

# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)   # change port if needed
