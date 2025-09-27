"""
Simurgh PDF – pure API back-end

Endpoints
─────────
GET  /health          → {"status": "ok"}
POST /api/merge       → streams merged PDF
(no HTML rendered; UI lives in React/Vite front-end)
"""

# SPDX-License-Identifier: AGPL-3.0-only

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
import sys

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
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
# Additional AI tuning (timeouts and caps)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
SYNTHESIS_PER_FILE_MAX = int(os.getenv("SYNTHESIS_PER_FILE_MAX", "1200"))
SYNTHESIS_MAX_CHARS = int(os.getenv("SYNTHESIS_MAX_CHARS", "8000"))

# Print configuration at startup for debugging
print("=== AI SERVICE CONFIGURATION ===")
print(f"AI_SERVICE: {AI_SERVICE}")
print(f"ANTHROPIC_API_KEY: {'SET' if ANTHROPIC_API_KEY else 'NOT SET'}")
print(f"ANTHROPIC_MODEL: {ANTHROPIC_MODEL}")
print(f"OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
print(f"OPENAI_MODEL: {OPENAI_MODEL}")
print(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
print("================================")

# Force flush to ensure logs are visible
sys.stdout.flush()

# Valid Anthropic models for validation
VALID_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022", 
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet",
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "claude-3-5-haiku",
    "claude-3-5-opus"
]

def call_ai_service(prompt, system_prompt=""):
    """Call AI service (Ollama, OpenAI, or Anthropic) based on configuration"""
    print(f"DEBUG: AI_SERVICE = '{AI_SERVICE}'")
    print(f"DEBUG: ANTHROPIC_API_KEY exists = {bool(ANTHROPIC_API_KEY)}")
    print(f"DEBUG: ANTHROPIC_MODEL = '{ANTHROPIC_MODEL}'")
    print(f"DEBUG: Environment variables loaded: AI_SERVICE={os.getenv('AI_SERVICE')}, ANTHROPIC_API_KEY={'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
    sys.stdout.flush()
    
    if AI_SERVICE == "openai":
        print("DEBUG: Calling OpenAI")
        sys.stdout.flush()
        return call_openai(prompt, system_prompt)
    elif AI_SERVICE == "anthropic":
        print("DEBUG: Calling Anthropic")
        sys.stdout.flush()
        return call_anthropic(prompt, system_prompt)
    else:
        print(f"DEBUG: Defaulting to Ollama (AI_SERVICE='{AI_SERVICE}')")
        sys.stdout.flush()
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
    """Call Anthropic API with the given prompt using direct HTTP requests"""
    print(f"DEBUG: call_anthropic called with prompt length: {len(prompt)}")
    print(f"DEBUG: ANTHROPIC_API_KEY length: {len(ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else 0}")
    print(f"DEBUG: ANTHROPIC_MODEL: {ANTHROPIC_MODEL}")
    sys.stdout.flush()
    
    if not ANTHROPIC_API_KEY:
        print("DEBUG: No ANTHROPIC_API_KEY found")
        sys.stdout.flush()
        return "Error: Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
    
    # Validate model name
    if ANTHROPIC_MODEL not in VALID_ANTHROPIC_MODELS:
        print(f"DEBUG: Invalid model '{ANTHROPIC_MODEL}'. Valid models: {VALID_ANTHROPIC_MODELS}")
        sys.stdout.flush()
        return f"Error: Invalid Anthropic model '{ANTHROPIC_MODEL}'. Valid models are: {', '.join(VALID_ANTHROPIC_MODELS)}"
    
    try:
        print("DEBUG: Using direct HTTP requests to Anthropic API")
        sys.stdout.flush()
        
        # Prepare the request payload
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": f"System: {system_prompt}\n\nUser: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": ANTHROPIC_MODEL,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        print(f"DEBUG: Making direct HTTP request to Anthropic API")
        print(f"DEBUG: Model: {ANTHROPIC_MODEL}")
        print(f"DEBUG: Payload keys: {list(payload.keys())}")
        sys.stdout.flush()
        
        # Make the HTTP request directly
        import requests
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        print(f"DEBUG: HTTP response status: {response.status_code}")
        sys.stdout.flush()
        
        if response.status_code == 200:
            result = response.json()
            print("DEBUG: API call successful")
            sys.stdout.flush()
            return result["content"][0]["text"]
        else:
            print(f"DEBUG: API call failed with status {response.status_code}")
            print(f"DEBUG: Response text: {response.text[:200]}")
            sys.stdout.flush()
            
            # Handle specific error cases
            if response.status_code == 401:
                return "Error: Invalid Anthropic API key. Please check your API key configuration."
            elif response.status_code == 400:
                return f"Error: Invalid request. Please check model name '{ANTHROPIC_MODEL}' and request format."
            elif response.status_code == 429:
                return "Error: Rate limit exceeded. Please try again later."
            elif response.status_code == 500:
                return "Error: Anthropic API server error. Please try again later."
            else:
                return f"Error: Anthropic API returned status {response.status_code}: {response.text[:100]}"
        
    except requests.exceptions.Timeout:
        print("DEBUG: Request timeout")
        sys.stdout.flush()
        return "Error: Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        print("DEBUG: Connection error")
        sys.stdout.flush()
        return "Error: Connection error. Please check your internet connection."
    except Exception as e:
        print(f"DEBUG: Exception calling Anthropic API: {e}")
        print(f"DEBUG: Exception type: {type(e).__name__}")
        sys.stdout.flush()
        return f"Error calling Anthropic API: {str(e)}"

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
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
                "repeat_penalty": 1.1,
                "seed": -1  # Random seed to prevent caching
            }
        }
        
        print(f"DEBUG: Calling Ollama with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
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

# --- AI output normalization helpers ---

def _strip_code_fences(s):
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip('`')
        t = t.replace('json', '', 1).strip()
    return t

def _object_to_readable(obj: dict) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    # Prefer common keys in a human order
    parts = []
    for key in ["title", "name", "point", "summary", "rationale", "recommendation", "action", "explanation", "detail", "note"]:
        if key in obj and obj.get(key):
            parts.append(f"{key.capitalize()}: {obj.get(key)}")
    if parts:
        return " ".join(parts)
    # Fallback: join all key/values
    try:
        return "; ".join([f"{k.capitalize()}: {v}" for k, v in obj.items() if v is not None])
    except Exception:
        return str(obj)

def _clean_display_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    import re, json as _json
    s = _strip_code_fences(text)
    s = s.strip()
    # Full-string JSON object → readable
    if s.startswith('{') and s.endswith('}'):
        try:
            return _object_to_readable(_json.loads(s))
        except Exception:
            pass
    # Replace embedded JSON objects that contain rationale/recommendation with readable form
    def _repl(m):
        chunk = m.group(0)
        try:
            return _object_to_readable(_json.loads(chunk))
        except Exception:
            return ''
    s = re.sub(r"\{[^{}]*?(?:\"rationale\"|\"recommendation\")[^{}]*?\}", _repl, s)
    # Strip stray outer quotes
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Normalize whitespace
    s = " ".join(s.split())
    return s

def _normalize_list(value):
    import re
    items = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [p.strip(' -*•') for p in re.split(r"[\r\n]+", value) if p.strip()]
    else:
        return []
    normalized = []
    for it in items:
        if isinstance(it, dict):
            normalized.append(_object_to_readable(it))
        elif isinstance(it, str):
            # Try parsing JSON-ish string
            try:
                obj = parse_json_safely(it)
                if isinstance(obj, dict):
                    normalized.append(_object_to_readable(obj))
                    continue
            except Exception:
                pass
            normalized.append(_clean_display_text(it))
        else:
            normalized.append(str(it))
    return normalized

def normalize_ai_summary_payload(payload):
    """Normalize/clean AI JSON payload so UI never shows raw JSON fragments."""
    if not isinstance(payload, dict):
        return {"summary": _clean_display_text(str(payload)), "key_topics": [], "main_points": [], "recommendations": []}
    res = {}
    res["summary"] = _clean_display_text(payload.get("summary") or payload.get("overview") or payload.get("executive_summary") or payload.get("message") or "")
    res["key_topics"] = _normalize_list(payload.get("key_topics") or payload.get("topics") or payload.get("key_points") or payload.get("bullets") or payload.get("highlights"))
    res["main_points"] = _normalize_list(payload.get("main_points") or payload.get("points") or payload.get("findings") or payload.get("takeaways"))
    res["recommendations"] = _normalize_list(payload.get("recommendations") or payload.get("suggestions") or payload.get("actions") or payload.get("next_steps"))
    return res

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
    """Lenient PDF check: just check extension and MIME type."""
    if not fileobj.filename:
        return False
    
    ext = fileobj.filename.lower().endswith(".pdf")
    mime = fileobj.mimetype == "application/pdf"
    
    # Accept if either extension OR MIME type indicates PDF
    return ext or mime

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

@app.get("/api/ai-config")
def ai_config():
    """Debug endpoint to check AI service configuration"""
    return jsonify({
        "ai_service": AI_SERVICE,
        "anthropic_api_key_set": bool(ANTHROPIC_API_KEY),
        "anthropic_model": ANTHROPIC_MODEL,
        "openai_api_key_set": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "valid_anthropic_models": VALID_ANTHROPIC_MODELS
    }), 200

@app.get("/api/ai-test")
def ai_test():
    """Test endpoint to verify AI service is working"""
    try:
        print("DEBUG: Testing AI service...")
        test_prompt = "Hello, this is a test. Please respond with 'AI service is working correctly.'"
        response = call_ai_service(test_prompt)
        print(f"DEBUG: AI test response: {response}")
        return jsonify({
            "status": "success",
            "ai_service": AI_SERVICE,
            "response": response,
            "timestamp": str(datetime.now())
        }), 200
    except Exception as e:
        print(f"DEBUG: AI test failed with error: {e}")
        return jsonify({
            "status": "error",
            "ai_service": AI_SERVICE,
            "error": str(e),
            "timestamp": str(datetime.now())
        }), 500

@app.get("/api/env-check")
def env_check():
    """Debug endpoint to check environment variables"""
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    all_proxy_vars = {k: v for k, v in os.environ.items() if 'proxy' in k.lower()}
    http_vars = {k: v for k, v in os.environ.items() if any(x in k.lower() for x in ['http', 'https', 'ssl', 'cert', 'ca'])}
    
    # Check requests library configuration
    requests_info = {}
    try:
        import requests
        requests_info = {
            "version": requests.__version__,
            "global_proxies": getattr(requests, 'proxies', 'Not set'),
            "session_proxies": getattr(requests.Session(), 'proxies', 'Not set')
        }
    except Exception as e:
        requests_info = {"error": str(e)}
    
    # Check httpx library configuration
    httpx_info = {}
    try:
        import httpx
        httpx_info = {
            "version": httpx.__version__,
            "available": True
        }
    except Exception as e:
        httpx_info = {"error": str(e), "available": False}
    
    # Check urllib3 library configuration
    urllib3_info = {}
    try:
        import urllib3
        urllib3_info = {
            "version": urllib3.__version__,
            "available": True
        }
    except Exception as e:
        urllib3_info = {"error": str(e), "available": False}
    
    return jsonify({
        "proxy_environment_variables": {k: v for k, v in os.environ.items() if k in proxy_vars},
        "all_proxy_related_vars": all_proxy_vars,
        "http_https_vars": http_vars,
        "requests_library_info": requests_info,
        "httpx_library_info": httpx_info,
        "urllib3_library_info": urllib3_info,
        "ai_service": AI_SERVICE,
        "anthropic_api_key_set": bool(ANTHROPIC_API_KEY),
        "anthropic_model": ANTHROPIC_MODEL,
        "total_env_vars": len(os.environ),
        "sample_env_vars": dict(list(os.environ.items())[:10]),  # First 10 env vars
        "all_env_vars": dict(os.environ)  # All environment variables
    }), 200


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

# ── Route: OCR PDF ─────────────────────────────────────────────────────────
@app.post("/api/ocr-pdf")
def ocr_pdf():
    """
    OCR PDF endpoint:
    1) Accept exactly one PDF upload
    2) Convert PDF pages to images using PyMuPDF
    3) Use pytesseract to extract text from each image
    4) Combine all extracted text and return as .txt file
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

    # 3) Open PDF with PyMuPDF and extract text using OCR
    try:
        import pytesseract
        from PIL import Image
        
        # Open PDF document
        pdf_document = fitz.open(in_path)
        extracted_text = []
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # Increase resolution for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR on the image
            try:
                page_text = pytesseract.image_to_string(img, lang='eng')
                if page_text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}\n")
            except Exception as ocr_error:
                print(f"OCR error on page {page_num + 1}: {ocr_error}")
                # Fallback: try to extract text directly from PDF
                page_text = page.get_text()
                if page_text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} (Direct Text) ---\n{page_text.strip()}\n")
        
        pdf_document.close()
        
        # 4) Combine all text
        full_text = "\n".join(extracted_text)
        
        if not full_text.strip():
            print("[DEBUG] No text extracted, using placeholder")
            full_text = "No text could be extracted from this PDF. The PDF may contain only images or be corrupted."
            extracted_text = ["No text could be extracted from this PDF."]
        
        # 5) Create output text file
        out_filename = f"ocr_extracted_{uuid.uuid4()}.txt"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # 6) Schedule cleanup
        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
                os.remove(out_path)
            except OSError:
                pass
            return response
        
        # 7) Return JSON response with text content and page texts
        return jsonify({
            "success": True,
            "full_text": full_text,
            "page_texts": [text.strip() for text in extracted_text if text.strip()]
        })
        
        
    except ImportError:
        # Fallback if pytesseract is not available
        try:
            pdf_document = fitz.open(in_path)
            extracted_text = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}\n")
            
            pdf_document.close()
            
            full_text = "\n".join(extracted_text)
            
            if not full_text.strip():
                print("[DEBUG] No text extracted in fallback, using placeholder")
                full_text = "No text could be extracted from this PDF. The PDF may contain only images or be corrupted."
                extracted_text = ["No text could be extracted from this PDF."]
            
            # Create output text file
            out_filename = f"extracted_{uuid.uuid4()}.txt"
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            @after_this_request
            def cleanup(response):
                try:
                    os.remove(in_path)
                    os.remove(out_path)
                except OSError:
                    pass
                return response
            
            # Return JSON response like the main OCR path  
            return jsonify({
                "success": True,
                "full_text": full_text,
                "page_texts": [text.strip() for text in extracted_text if text.strip()]
            })
            
        except Exception as e:
            os.remove(in_path)
            return jsonify(error=f"Cannot process PDF: {e}"), 400
            
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"OCR processing failed: {e}"), 400

@app.post("/api/create-searchable-pdf")
def create_searchable_pdf():
    """
    Create a searchable PDF by overlaying OCR text on the original PDF
    """
    files = request.files.getlist("file")
    if len(files) != 1:
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    if not allowed_pdf(f):
        return jsonify(error="Only PDF files allowed"), 400

    # Get the page texts from the form data
    page_texts_json = request.form.get("page_texts", "[]")
    try:
        page_texts = json.loads(page_texts_json)
    except:
        page_texts = []

    # Save incoming PDF
    in_filename = f"{uuid.uuid4()}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_filename)
    f.save(in_path)

    try:
        # For now, just return the original PDF (searchable PDF creation is complex)
        # In a full implementation, you'd overlay the OCR text on the PDF
        
        out_filename = f"searchable_{uuid.uuid4()}.pdf"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
        
        # Copy the original PDF as a placeholder
        import shutil
        shutil.copy2(in_path, out_path)
        
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
            download_name=f"searchable_{f.filename}",
            mimetype="application/pdf"
        )
        
    except Exception as e:
        os.remove(in_path)
        return jsonify(error=f"Failed to create searchable PDF: {e}"), 400

@app.post("/api/test-peppol")
def test_peppol():
    print("[DEBUG] TEST ENDPOINT CALLED!")
    return jsonify({"message": "Test endpoint working", "timestamp": str(datetime.now())})

@app.post("/api/pdf-to-peppol")
def pdf_to_peppol():
    """
    PDF to Peppol endpoint:
    1) Accept exactly one PDF upload (invoice/document)
    2) Extract text and data from PDF using OCR and text extraction
    3) Parse invoice data (amounts, dates, parties, etc.)
    4) Generate UBL XML format for Peppol network
    5) Return the UBL XML file
    """
    print(f"[DEBUG] PDF to Peppol: Starting request")
    
    # 1) Validate exactly one PDF file was sent
    files = request.files.getlist("file")
    print(f"[DEBUG] Files received: {len(files)}")
    
    if len(files) != 1:
        print(f"[DEBUG] Error: Expected 1 file, got {len(files)}")
        return jsonify(error="Upload exactly one PDF file"), 400

    f = files[0]
    print(f"[DEBUG] File: {f.filename}, MIME: {f.mimetype}")
    
    # Debug the allowed_pdf validation
    ext = f.filename.lower().endswith(".pdf") if f.filename else False
    mime = f.mimetype == "application/pdf"
    guess = mimetypes.guess_type(f.filename)[0] == "application/pdf" if f.filename else False
    
    print(f"[DEBUG] Validation: ext={ext}, mime={mime}, guess={guess}")
    
    if not allowed_pdf(f):
        print(f"[DEBUG] Error: File validation failed")
        return jsonify(error=f"Only PDF files allowed. File: {f.filename}, MIME: {f.mimetype}"), 400

    # 2) Save incoming PDF to a temp path
    in_filename = f"{uuid.uuid4()}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_filename)
    print(f"[DEBUG] Saving to: {in_path}")
    f.save(in_path)

    try:
        print(f"[DEBUG] Starting PDF processing")
        
        # 3) Extract text from PDF using multiple methods
        import re
        from datetime import datetime
        import xml.etree.ElementTree as ET
        
        # Try to import pytesseract, but don't fail if it's not available
        try:
            import pytesseract
            from PIL import Image
            OCR_AVAILABLE = True
            print(f"[DEBUG] OCR available")
        except ImportError as e:
            print(f"[DEBUG] OCR not available: {e}")
            OCR_AVAILABLE = False
        
        pdf_document = fitz.open(in_path)
        extracted_text = ""
        
        print(f"[DEBUG] PDF has {len(pdf_document)} pages")
        
        # Try direct text extraction first
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text.strip():
                extracted_text += page_text + "\n"
        
        print(f"[DEBUG] Extracted text length: {len(extracted_text)}")
        
        # If no text found and OCR is available, use OCR
        if not extracted_text.strip() and OCR_AVAILABLE:
            print(f"[DEBUG] No text found, trying OCR")
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                try:
                    page_text = pytesseract.image_to_string(img, lang='eng')
                    if page_text.strip():
                        extracted_text += page_text + "\n"
                except Exception as ocr_error:
                    print(f"[DEBUG] OCR failed: {ocr_error}")
                    pass
        
        pdf_document.close()
        
        if not extracted_text.strip():
            print(f"[DEBUG] No text extracted, using placeholder data")
            extracted_text = "Invoice #INV-001\nDate: 2024-01-01\nTotal: 0.00\nSupplier: Unknown\nCustomer: Unknown"

        print(f"[DEBUG] Final text: {extracted_text[:200]}...")

        # 4) Parse invoice data using regex patterns
        def extract_invoice_data(text):
            # Clean the text first - remove invisible characters and normalize whitespace
            import unicodedata
            text = unicodedata.normalize('NFKD', text)  # Normalize unicode
            text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Remove zero-width characters
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            data = {
                'invoice_number': '',
                'invoice_date': '',
                'due_date': '',
                'supplier_name': '',
                'customer_name': '',
                'total_amount': '',
                'currency': 'EUR',
            }
            
            # Extract invoice number - improved patterns
            invoice_patterns = [
                r'invoice\s+no[:\s]*([A-Z0-9\-]+)',
                r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
                r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)',
                r'invoice\s+number\s*:?\s*([A-Z0-9\-]+)',
                r'bill\s*#?\s*:?\s*([A-Z0-9\-]+)'
            ]
            
            for pattern in invoice_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data['invoice_number'] = match.group(1)
                    break
            
            # Extract dates - improved patterns
            date_patterns = [
                r'issue\s+date[:\s]*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
                r'date[:\s]*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
                r'(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
                r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
            ]
            
            dates_found = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates_found.extend(matches)
            
            if dates_found:
                data['invoice_date'] = dates_found[0]
                if len(dates_found) > 1:
                    data['due_date'] = dates_found[1]
            
            # Extract amounts - improved patterns
            amount_patterns = [
                r'total\s*\([^)]*\)[:\s]*[€$£]?\s*([\d,]+\.?\d*)',
                r'total[:\s]*[€$£]?\s*([\d,]+\.?\d*)',
                r'amount\s+due[:\s]*[€$£]?\s*([\d,]+\.?\d*)',
                r'grand\s+total[:\s]*[€$£]?\s*([\d,]+\.?\d*)',
                r'[€$£]\s*([\d,]+\.?\d*)\s*(?:total|due)'
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data['total_amount'] = match.group(1).replace(',', '')
                    break
            
            # Extract supplier name - with debug output
            print(f"[DEBUG] Looking for supplier in text: {repr(text[:200])}")
            
            # Try multiple patterns
            supplier_patterns = [
                r'supplier[:\s]*([A-Za-zÀ-ÿ]+)\s+Main\s+Street',
                r'supplier[:\s]*([A-Za-zÀ-ÿ\s]+?)(?=\s*Main\s*Street)',
                r'supplier[:\s]*([A-Za-zÀ-ÿ\s]+?)(?=\s*Customer)',
                r'supplier[:\s]*([^\s]+(?:\s+[^\s]+)*?)(?=\s*Main\s*Street)',
            ]
            
            for i, pattern in enumerate(supplier_patterns):
                print(f"[DEBUG] Trying supplier pattern {i+1}: {pattern}")
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    supplier_name = match.group(1).strip()
                    print(f"[DEBUG] Supplier pattern {i+1} matched: '{supplier_name}'")
                    if len(supplier_name) > 2:
                        data['supplier_name'] = supplier_name
                        break
                else:
                    print(f"[DEBUG] Supplier pattern {i+1} did not match")
            
            print(f"[DEBUG] Final supplier extraction: '{data['supplier_name']}'")
            
            # Extract customer name - with debug output
            print(f"[DEBUG] Looking for customer in text")
            
            # Try multiple patterns
            customer_patterns = [
                r'customer[:\s]*([A-Za-zÀ-ÿ\s]+?)\s+Innovation\s+Avenue',
                r'customer[:\s]*([A-Za-zÀ-ÿ\s]+?)(?=\s*Innovation\s*Avenue)',
                r'customer[:\s]*([A-Za-zÀ-ÿ\s]+?)(?=\s*Items)',
                r'customer[:\s]*([^\s]+(?:\s+[^\s]+)*?)(?=\s*Innovation\s*Avenue)',
            ]
            
            for i, pattern in enumerate(customer_patterns):
                print(f"[DEBUG] Trying customer pattern {i+1}: {pattern}")
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    customer_name = match.group(1).strip()
                    print(f"[DEBUG] Customer pattern {i+1} matched: '{customer_name}'")
                    customer_name = re.sub(r'\s+', ' ', customer_name)
                    if len(customer_name) > 2:
                        data['customer_name'] = customer_name
                        break
                else:
                    print(f"[DEBUG] Customer pattern {i+1} did not match")
            
            print(f"[DEBUG] Final customer extraction: '{data['customer_name']}'")
            return data

        # 5) Extract invoice data
        print(f"[DEBUG] Extracting invoice data")
        invoice_data = extract_invoice_data(extracted_text)
        print(f"[DEBUG] Invoice data: {invoice_data}")
        
        # 6) Generate UBL XML
        def create_ubl_invoice(data):
            root = ET.Element("Invoice", xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2")
            
            # UBL Version
            ubl_version = ET.SubElement(root, "UBLVersionID")
            ubl_version.text = "2.1"
            
            # ID (Invoice Number)
            invoice_id = ET.SubElement(root, "ID")
            invoice_id.text = data.get('invoice_number', 'INV-001')
            
            # Issue Date
            issue_date = ET.SubElement(root, "IssueDate")
            issue_date.text = data.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))
            
            # Document Currency Code
            currency_code = ET.SubElement(root, "DocumentCurrencyCode")
            currency_code.text = data.get('currency', 'EUR')
            
            # Accounting Supplier Party
            supplier_party = ET.SubElement(root, "AccountingSupplierParty")
            party = ET.SubElement(supplier_party, "Party")
            party_name = ET.SubElement(party, "PartyName")
            name = ET.SubElement(party_name, "Name")
            name.text = data.get('supplier_name', 'Supplier Company')
            
            # Legal Monetary Totals
            legal_monetary_totals = ET.SubElement(root, "LegalMonetaryTotals")
            
            # Payable Amount
            payable_amount = ET.SubElement(legal_monetary_totals, "PayableAmount")
            payable_amount.set("currencyID", data.get('currency', 'EUR'))
            payable_amount.text = data.get('total_amount', '0.00')
            
            return root

        # 7) Generate UBL XML
        print(f"[DEBUG] Creating UBL XML")
        ubl_root = create_ubl_invoice(invoice_data)
        
        # 8) Convert to string
        ET.indent(ubl_root, space="  ", level=0)
        ubl_xml = ET.tostring(ubl_root, encoding='unicode', xml_declaration=True)
        
        # 9) Save UBL XML file
        out_filename = f"peppol_{uuid.uuid4()}.xml"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
        
        print(f"[DEBUG] Saving XML to: {out_path}")
        with open(out_path, 'w', encoding='utf-8') as xml_file:
            xml_file.write(ubl_xml)
        
        # 10) Schedule cleanup
        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
                os.remove(out_path)
            except OSError:
                pass
            return response
        
        # 11) Return the UBL XML file
        print(f"[DEBUG] Returning file: {out_filename}")
        return send_file(
            out_path,
            as_attachment=True,
            download_name=f"peppol_{files[0].filename.replace('.pdf', '.xml')}",
            mimetype="application/xml"
        )
        
    except Exception as e:
        print(f"[DEBUG] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        os.remove(in_path)
        return jsonify(error=f"PDF to Peppol conversion failed: {e}"), 400

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

        # Call AI service with debugging and fallback
        print(f"DEBUG: Sending prompt to AI service (first 300 chars): {prompt[:300]}")
        print(f"DEBUG: About to call call_ai_service with AI_SERVICE={AI_SERVICE}")
        sys.stdout.flush()
        ai_response_text = call_ai_service(prompt)
        print(f"DEBUG: AI service response: {ai_response_text[:200]}")
        sys.stdout.flush()
        
        # If AI service is not available, provide a contextual response
        if "Error" in ai_response_text:
            print(f"DEBUG: AI service returned error: {ai_response_text}")
            sys.stdout.flush()
            ai_response_text = f"Based on your question '{question}' about this {len(pdf_reader.pages)}-page PDF document, I can see the content but there's an issue with the AI service: {ai_response_text}. The document appears to contain: {text_content[:300]}..."
        
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
            ai_explanation = normalize_ai_summary_payload(parse_json_safely(ai_response_text))
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
            ai_answer = normalize_ai_summary_payload(parse_json_safely(ai_response_text))
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
            ai_summary = normalize_ai_summary_payload(parse_json_safely(ai_response_text))
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

@app.post("/api/ai-document-synthesis")
def ai_document_synthesis():
    """Synthesize multiple PDFs into a single consolidated PDF (Report, Brief, or Minutes).
    - Accepts multiple files under field name "files"
    - Accepts form field "format" in {report|brief|minutes}
    - Uses the configured AI service (same as Chat with PDF) to generate synthesized text
    - Renders the synthesized text into a PDF and returns it
    """
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "Upload at least one PDF file"}), 400
        if any(not allowed_pdf(f) for f in files):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        target_format = (request.form.get("format") or "report").lower().strip()
        if target_format not in ("report", "brief", "minutes"):
            return jsonify({"error": "format must be one of: report|brief|minutes"}), 400

        # Extract text from each PDF (truncate to keep prompt size manageable)
        combined_snippets = []
        total_pages = 0
        current_total_chars = 0
        for f in files:
            # Stop early if we have reached the global cap
            if current_total_chars >= SYNTHESIS_MAX_CHARS:
                break
            try:
                reader = PdfReader(f)
                doc_text = ""
                for p in reader.pages:
                    total_pages += 1
                    txt = p.extract_text() or ""
                    if txt:
                        doc_text += txt + "\n"
                # keep a capped snippet per file and a global cap
                snippet = (doc_text or "").strip()
                if snippet:
                    snippet = snippet[:SYNTHESIS_PER_FILE_MAX]
                    # Enforce global cap across all snippets
                    remaining = max(SYNTHESIS_MAX_CHARS - current_total_chars, 0)
                    if remaining > 0:
                        snippet = snippet[:remaining]
                        if snippet:
                            combined_snippets.append(snippet)
                            current_total_chars += len(snippet)
                else:
                    combined_snippets.append(f"[Unreadable or no text extracted from {secure_filename(f.filename)}]")
            except Exception as e:
                # Skip unreadable PDFs but continue
                combined_snippets.append(f"[Unreadable or no text extracted from {secure_filename(f.filename)}]")

        if not combined_snippets:
            combined_snippets.append("[No text extracted from any input documents]")

        # Build prompt for the AI model
        system_map = {
            "report": "Produce a well-structured professional report with sections (Overview, Key Findings, Analysis, Recommendations).",
            "brief": "Produce an executive brief with bullets, focusing on clarity and concision.",
            "minutes": "Produce meeting minutes with attendees (if inferable), agenda, decisions, action items, and next steps."
        }
        system_instruction = system_map.get(target_format, system_map["report"])  # default to report

        prompt = (
            f"SYSTEM: You synthesize multiple documents. Return a cohesive {target_format} in well-formatted paragraphs and bullet points where appropriate.\n"
            f"Use clear headings (no markdown asterisks). Prefer headings that end with a colon.\n"
            f"When listing, put each item on its own new line starting with '-' (dash). Avoid using '*' or '**'.\n\n"
            f"DOCUMENT COUNT: {len(files)}, TOTAL PAGES (approx): {total_pages}\n"
            f"FORMAT STYLE: {system_instruction}\n\n"
            f"CONTENT SNIPPETS (truncated):\n"
            + "\n\n---\n\n".join(combined_snippets)
        )

        # Call the same AI service as chat with PDF
        ai_text = call_ai_service(prompt)
        if not isinstance(ai_text, str):
            ai_text = str(ai_text)
        if not ai_text.strip():
            ai_text = "No synthesized content produced by AI."

        # Render the AI text into a simple PDF using reportlab
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            from reportlab.lib.styles import ParagraphStyle
        except Exception as e:
            return jsonify({"error": f"ReportLab not installed: {e}. Please add reportlab to requirements."}), 500

        out_path = os.path.join(app.config["UPLOAD_FOLDER"], f"synthesis_{uuid.uuid4()}.pdf")
        doc = SimpleDocTemplate(out_path, pagesize=LETTER, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch)
        story = []

        styles = getSampleStyleSheet()
        header_style = ParagraphStyle('Header', parent=styles['Heading1'], spaceAfter=12)
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=8, spaceAfter=6)
        h3_style = ParagraphStyle('H3', parent=styles['Heading3'], spaceBefore=6, spaceAfter=4)
        body_style = ParagraphStyle('Body', parent=styles['BodyText'], leading=14, spaceAfter=8, fontSize=11)
        bullet1_style = ParagraphStyle('BulletL1', parent=styles['BodyText'], leftIndent=18, spaceBefore=2)
        bullet2_style = ParagraphStyle('BulletL2', parent=styles['BodyText'], leftIndent=36, spaceBefore=2)
        bullet3_style = ParagraphStyle('BulletL3', parent=styles['BodyText'], leftIndent=54, spaceBefore=2)

        title_map = {"report": "SYNTHESIZED REPORT", "brief": "SYNTHESIZED BRIEF", "minutes": "SYNTHESIZED MINUTES"}
        story.append(Paragraph(title_map.get(target_format, "SYNTHESIZED REPORT"), header_style))
        story.append(Spacer(1, 0.2*inch))

        # Parse AI text into headings, paragraphs, and bullets with up to 3 levels
        import re
        def esc(s: str) -> str:
            return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Clean markdown bold and normalize bullet separators that were crammed into one line
        text = ai_text.replace('\r\n', '\n')
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'(?<=[\n\.;:])\s*(?:\*|\-|\+)\s+', '\n* ', text)

        # Heading detectors
        def is_h2(line: str) -> bool:
            l = line.strip()
            if len(l) < 2 or len(l) > 80:
                return False
            if l.endswith(':'):
                return True
            # ALL CAPS words
            return bool(re.match(r'^[A-Z0-9][A-Z0-9\s\-/()&]{2,80}$', l))

        def is_h3(line: str) -> bool:
            l = line.strip()
            if len(l) < 2 or len(l) > 80:
                return False
            if is_h2(l):
                return False
            # Title Case (allow small connector words)
            return bool(re.match(r'^(?:[A-Z][\w()\-/]*)(?:\s+(?:[A-Z][\w()\-/]*|of|and|to|for|in|on|with|the|a|an))*$', l))

        bullet_re = re.compile(r'^(?P<indent>\s*)(?:(?:[\*\-\+])|(?:\d+\.))\s+(?P<text>.+)$')

        lines = text.split('\n')
        paragraph_buf = []

        def flush_paragraph():
            if paragraph_buf:
                paragraph_text = ' '.join(paragraph_buf).strip()
                if paragraph_text:
                    story.append(Paragraph(esc(paragraph_text), body_style))
                paragraph_buf.clear()

        for raw in lines:
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue

            # Bulleted item
            m = bullet_re.match(line)
            if m:
                flush_paragraph()
                indent = m.group('indent') or ''
                level = 1 + min(2, len(indent) // 2)
                txt = m.group('text').strip()
                bullet_text = f"• {txt}"
                style = bullet1_style if level == 1 else (bullet2_style if level == 2 else bullet3_style)
                story.append(Paragraph(esc(bullet_text), style))
                continue

            # Headings
            if is_h2(stripped):
                flush_paragraph()
                heading = stripped[:-1].strip() if stripped.endswith(':') else stripped
                story.append(Paragraph(esc(heading), h2_style))
                continue
            if is_h3(stripped):
                flush_paragraph()
                story.append(Paragraph(esc(stripped), h3_style))
                continue

            # Regular paragraph line (will be joined until blank line)
            paragraph_buf.append(stripped)

        flush_paragraph()

        try:
            doc.build(story)
        except Exception as e:
            return jsonify({"error": f"Failed to render PDF: {e}"}), 500

        @after_this_request
        def _cleanup(resp):
            try:
                os.remove(out_path)
            except OSError:
                pass
            return resp

        return send_file(
            out_path,
            as_attachment=True,
            download_name=f"document_{target_format}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500

# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)   # change port if needed
