"""
Summarize service orchestrator: coordinates the entire summarize pipeline.
"""
import os
import sys
from typing import Dict, Any, List

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.pdf_utils import extract_text_from_pdf, extract_text_by_page, compute_text_density
from common.llm_client import LLMClient
from common.metrics import JobMetrics
from summarize.prompt_pack import build_summarize_prompt
from summarize.normalizer import normalize_summarize_result
from summarize.provenance import map_summary_to_provenance
from summarize.quality import evaluate_quality


class SummarizeService:
    """Service to summarize PDF documents with domain intelligence."""
    
    def __init__(self):
        # Use AI_SERVICE env var or default to ollama
        ai_service = os.getenv("AI_SERVICE", "ollama").lower()
        model = None
        if ai_service == "ollama":
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        elif ai_service == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4")
        elif ai_service == "anthropic":
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        
        self.llm_client = LLMClient(provider=ai_service, model=model)
    
    def process(self, pdf_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            params: {domain, detail, provenance, custom_instructions, use_ocr, extract_tables}
        
        Returns:
            {result: {}, quality: {}, metrics: {}}
        """
        metrics = JobMetrics()
        
        domain = params.get("domain", "general")
        detail = params.get("detail", "executive")
        provenance_enabled = params.get("provenance", False)
        custom_instructions = params.get("custom_instructions", "")
        use_ocr = params.get("use_ocr", False)
        extract_tables = params.get("extract_tables", False)
        
        # Stage 1: Text extraction
        metrics.mark_stage("text_extraction_start")
        
        text_density = compute_text_density(pdf_path)
        needs_ocr = use_ocr or text_density < 100
        
        if needs_ocr:
            try:
                from ..extraction.ocr_service import OCRService
                ocr_service = OCRService()
                full_text = ocr_service.extract_text(pdf_path)
                source_pages = ocr_service.extract_text_by_page(pdf_path)
            except Exception as e:
                full_text = extract_text_from_pdf(pdf_path)
                source_pages = extract_text_by_page(pdf_path)
        else:
            full_text = extract_text_from_pdf(pdf_path)
            source_pages = extract_text_by_page(pdf_path)
        
        metrics.mark_stage("text_extraction_done")
        
        # Stage 2: Optional table extraction
        tables = []
        if extract_tables:
            try:
                from ..extraction.plugins.table_extractor import TableExtractor
                table_extractor = TableExtractor()
                tables = table_extractor.extract(pdf_path)
            except:
                pass
        
        metrics.mark_stage("table_extraction_done")
        
        # Stage 3: Build prompt
        prompt = build_summarize_prompt(full_text, domain, detail, provenance_enabled, custom_instructions, tables)
        metrics.mark_stage("prompt_built")
        
        # Stage 4: LLM call
        llm_response = self.llm_client.call(prompt)
        metrics.add_llm_call(llm_response["tokens"], llm_response.get("cost", 0.0))
        metrics.mark_stage("llm_done")
        
        # Stage 5: Normalize
        result = normalize_summarize_result(llm_response["text"])
        metrics.mark_stage("normalization_done")
        
        # Stage 6: Provenance mapping
        provenance = {}
        if provenance_enabled:
            provenance = map_summary_to_provenance(result, source_pages, full_text)
            result["provenance"] = provenance
        metrics.mark_stage("provenance_done")
        
        # Stage 7: Quality evaluation
        quality = evaluate_quality(result, domain, provenance)
        metrics.mark_stage("quality_done")
        
        metrics.finish()
        
        return {
            "result": result,
            "quality": quality,
            "metrics": metrics.to_dict()
        }

