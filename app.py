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
import mimetypes, os, uuid
import subprocess, shlex          # run Ghostscript
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

if os.getenv("FLASK_ENV") == "development":
    origins.append("http://localhost:5173")

# PRODUCTION CORS: only front-end domains may access /api/*
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "https://perkpdf.com",
        "https://www.perkpdf.com"
    ]}}
)

# ── config & housekeeping ───────────────────────────────────────
app.config["UPLOAD_FOLDER"]      = "temp"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024    # 25 MB
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

# ── ROUTES ───────────────────────────────────────────────────────
@app.get("/")
def root():
    """Simple root for anyone hitting the API directly."""
    return {"service": "Simurgh PDF API", "docs": "/health"}, 200

@app.get("/health")
def health():
    """Used by React (and uptime checks) to verify API is alive."""
    return jsonify(status="ok"), 200


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


# ── SPLIT (NEW) ───────────────────────────────────────────
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


# ── COMPRESS (NEW) ────────────────────────
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

    # Ghostscript command
    gs_cmd = (
        f"gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 "
        f"-dPDFSETTINGS=/{quality} -dNOPAUSE -dQUIET -dBATCH "
        f"-sOutputFile={shlex.quote(out_path)} {shlex.quote(in_path)}"
    )

    try:
        subprocess.run(shlex.split(gs_cmd), check=True)
    except subprocess.CalledProcessError:
        return jsonify(error="Compression failed"), 500

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

# ── run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)   # change port if needed
