"""
Flask endpoints for Summarize tool.
"""
import os
import sys
import uuid
import threading
from flask import request, jsonify
from werkzeug.utils import secure_filename

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarize.jobs import create_job, update_job, get_job, set_job_result, set_job_error
from summarize.service import SummarizeService


def register_summarize_endpoints(app):
    """Register Summarize endpoints with Flask app."""
    
    @app.post("/api/ai-summarize-pdf/async")
    def summarize_pdf_async():
        """Async summarize endpoint - returns job_id immediately."""
        try:
            # Validate file
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '' or not file.filename.endswith('.pdf'):
                return jsonify({"error": "Invalid PDF file"}), 400
            
            # Get parameters
            domain = request.form.get('domain', 'general')
            detail = request.form.get('detail', 'executive')
            provenance = request.form.get('provenance', 'false').lower() == 'true'
            custom_instructions = request.form.get('custom_instructions', '')
            use_ocr = request.form.get('use_ocr', 'false').lower() == 'true'
            extract_tables = request.form.get('extract_tables', 'false').lower() == 'true'
            
            # Save file to temp
            job_id = str(uuid.uuid4())
            temp_dir = app.config.get("UPLOAD_FOLDER", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join(temp_dir, f"{job_id}_{filename}")
            file.save(temp_path)
            
            # Create job
            params = {
                "domain": domain,
                "detail": detail,
                "provenance": provenance,
                "custom_instructions": custom_instructions,
                "use_ocr": use_ocr,
                "extract_tables": extract_tables,
                "filename": filename
            }
            create_job(job_id, params)
            
            # Start background processing
            def process():
                try:
                    update_job(job_id, {"status": "processing", "progress": 10})
                    
                    service = SummarizeService()
                    
                    update_job(job_id, {"progress": 30})
                    
                    output = service.process(temp_path, params)
                    
                    update_job(job_id, {"progress": 90})
                    
                    # Store result
                    set_job_result(job_id, output["result"], output["metrics"])
                    
                    # Add quality to result
                    job = get_job(job_id)
                    if job and job.get("result"):
                        job["result"]["quality"] = output["quality"]
                        update_job(job_id, {"result": job["result"]})
                    
                except Exception as e:
                    set_job_error(job_id, str(e))
                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
            
            thread = threading.Thread(target=process, daemon=True)
            thread.start()
            
            return jsonify({"job_id": job_id}), 202
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.get("/api/ai-summarize-pdf/status/<job_id>")
    def summarize_pdf_status(job_id: str):
        """Get status of summarize job."""
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error"),
            "done": job["status"] in ("completed", "failed")
        })
    
    @app.get("/api/ai-summarize-pdf/result/<job_id>")
    def summarize_pdf_result(job_id: str):
        """Get result of completed summarize job."""
        job = get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        if job["status"] != "completed":
            return jsonify({"error": "Job not completed"}), 400
        
        return jsonify(job["result"])

