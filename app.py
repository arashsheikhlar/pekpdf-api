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
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

origins = []
if os.getenv("FLASK_ENV") == "development":
    origins.append("http://localhost:5173")

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
    If you’d prefer a single image instead of a ZIP, you could adapt this code.
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

            # Write each page’s PNG into the ZIP as page_1.png, page_2.png, etc.
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


# ── run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)   # change port if needed
