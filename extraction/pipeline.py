# SPDX-License-Identifier: AGPL-3.0-only

"""
Main extraction pipeline.

This module orchestrates all extraction services to provide a unified
interface for document extraction with caching, AI processing, and plugin support.
"""

from typing import Optional, Dict, Any, List
import os

from .models import ExtractionOptions, ExtractionResult
from .domain_service import DomainService
from .ai_service import AIExtractionService
from .text_service import TextExtractionService
from .cache_service import CacheService
from .confidence import ConfidenceScorer
from .plugins import get_plugin_registry
from .config import config


class ExtractionPipeline:
    """Main pipeline for document extraction."""
    
    def __init__(
        self,
        domain_service: DomainService,
        ai_service: AIExtractionService,
        text_service: TextExtractionService,
        cache_service: CacheService,
        confidence_scorer: ConfidenceScorer
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            domain_service: Service for domain management
            ai_service: Service for AI-powered extraction
            text_service: Service for text extraction
            cache_service: Service for caching results
            confidence_scorer: Service for confidence scoring
        """
        self.domain_service = domain_service
        self.ai_service = ai_service
        self.text_service = text_service
        self.cache_service = cache_service
        self.confidence_scorer = confidence_scorer
        self.plugin_registry = get_plugin_registry()
    
    def extract(self, file_path: str, options: ExtractionOptions) -> ExtractionResult:
        """
        Extract information from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            options: Extraction options
            
        Returns:
            Extraction result with all extracted data
        """
        # Check cache first
        if config.cache_enabled:
            cache_key = self._compute_cache_key(file_path, options)
            cached_result = self.cache_service.get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Extract text from PDF
        all_text, pages_text = self.text_service.extract_text_from_pdf(
            file_path, options.use_ocr
        )
        
        if not all_text.strip():
            # Return empty result if no text extracted
            return ExtractionResult(
                type="general",
                pages=len(pages_text),
                entities={},
                mapped_fields={},
                validation={"errors": ["No text could be extracted from the document"], "warnings": []},
                confidence={"overall": 0, "fields": {}}
            )
        
        # Run AI extraction
        ai_result = self.ai_service.extract_document(all_text, pages_text, options)
        
        # Run plugins for additional extraction
        plugin_results = self._run_plugins(all_text, pages_text, options, ai_result.type)
        
        # Merge plugin results
        merged_result = self._merge_plugin_results(ai_result, plugin_results)
        
        # Compute confidence scores (convert Pydantic models to dicts)
        confidence = self.confidence_scorer.score(
            merged_result.type,
            merged_result.mapped_fields.model_dump() if hasattr(merged_result.mapped_fields, 'model_dump') else merged_result.mapped_fields,
            merged_result.entities.model_dump() if hasattr(merged_result.entities, 'model_dump') else merged_result.entities,
            merged_result.validation.model_dump() if hasattr(merged_result.validation, 'model_dump') else merged_result.validation,
            merged_result.provenance.model_dump() if hasattr(merged_result.provenance, 'model_dump') else merged_result.provenance,
            options.selected_fields
        )
        
        # Update confidence in result
        merged_result.confidence = confidence
        
        # Optional enrichment
        if options.enrich:
            enriched = self.ai_service.enrich_extraction(
                merged_result.type,
                merged_result.mapped_fields,
                all_text,
                pages_text,
                options
            )
            merged_result.mapped_fields = enriched.get('mapped_fields', merged_result.mapped_fields)
            
            # Re-compute confidence after enrichment
            if merged_result.mapped_fields.get('enriched'):
                confidence = self.confidence_scorer.score(
                    merged_result.type,
                    merged_result.mapped_fields,
                    merged_result.entities,
                    merged_result.validation,
                    merged_result.provenance,
                    options.selected_fields
                )
                merged_result.confidence = confidence
        
        # Cache result
        if config.cache_enabled:
            self.cache_service.cache_result(cache_key, merged_result)
        
        return merged_result
    
    def extract_with_provenance(self, file_path: str, options: ExtractionOptions) -> ExtractionResult:
        """
        Extract information with detailed provenance for research documents.
        
        Args:
            file_path: Path to the PDF file
            options: Extraction options
            
        Returns:
            Extraction result with provenance information
        """
        # Get basic extraction result
        result = self.extract(file_path, options)
        
        # Add provenance for research documents
        if result.type == "research":
            result.provenance = self._build_research_provenance(result, file_path)
        
        return result
    
    def _compute_cache_key(self, file_path: str, options: ExtractionOptions) -> str:
        """Compute cache key for file and options."""
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            return self.cache_service.compute_cache_key(file_bytes, options)
        except Exception:
            # Fallback to file path hash if file reading fails
            import hashlib
            return hashlib.md5(f"{file_path}_{options.dict()}".encode()).hexdigest()
    
    def _run_plugins(self, text: str, pages_text: List[str], options: ExtractionOptions, document_type: str) -> List[Dict[str, Any]]:
        """Run applicable plugins for additional extraction."""
        plugin_results = []
        
        if not config.plugins_enabled:
            return plugin_results
        
        # Find applicable plugins
        applicable_plugins = self.plugin_registry.find_applicable_plugins(document_type, options)
        
        # Run each plugin based on options
        for plugin in applicable_plugins:
            plugin_name = plugin.get_name()
            
            # Check if this plugin should run based on options
            if plugin_name == "table_extractor" and not options.extract_tables:
                continue
            if plugin_name == "formula_extractor" and not options.extract_formulas:
                continue
            
            try:
                result = plugin.extract(text, pages_text, options)
                plugin_results.append({
                    'plugin': plugin_name,
                    'result': result
                })
            except Exception as e:
                # Log error but continue with other plugins
                print(f"Plugin {plugin_name} failed: {e}")
        
        return plugin_results
    
    def _merge_plugin_results(self, ai_result: ExtractionResult, plugin_results: List[Dict[str, Any]]) -> ExtractionResult:
        """Merge plugin results into the main extraction result."""
        merged_result = ai_result
        
        # Merge tables from plugins
        all_tables = list(ai_result.tables)
        for plugin_result in plugin_results:
            if 'tables' in plugin_result['result']:
                all_tables.extend(plugin_result['result']['tables'])
        
        merged_result.tables = all_tables
        
        # Merge formulas from plugins
        all_formulas = list(ai_result.formulas)
        for plugin_result in plugin_results:
            if 'formulas' in plugin_result['result']:
                all_formulas.extend(plugin_result['result']['formulas'])
        
        merged_result.formulas = all_formulas
        
        return merged_result
    
    def _build_research_provenance(self, result: ExtractionResult, file_path: str) -> Dict[str, Any]:
        """Build provenance information for research documents."""
        provenance = {}
        
        try:
            # Build inverted index for page references
            inv_index = self.text_service.build_inverted_index(result.pages_text if hasattr(result, 'pages_text') else [])
            
            # Add page references for key fields
            for field_name, field_value in result.mapped_fields.items():
                if isinstance(field_value, str) and field_value.strip():
                    # Find pages containing this field value
                    pages = self._find_pages_for_value(field_value, inv_index)
                    if pages:
                        provenance[f"{field_name}_pages"] = pages[:10]  # Limit to 10 pages
            
            # Add general provenance metadata
            provenance['extraction_method'] = 'ai_with_provenance'
            provenance['file_path'] = os.path.basename(file_path)
            
        except Exception:
            # Return minimal provenance if building fails
            provenance = {'extraction_method': 'ai_basic'}
        
        return provenance
    
    def _find_pages_for_value(self, value: str, inv_index: Dict[str, List[int]]) -> List[int]:
        """Find pages containing a specific value."""
        pages = set()
        words = value.lower().split()
        
        for word in words:
            if len(word) >= 4 and word in inv_index:
                pages.update(inv_index[word])
        
        return sorted(list(pages))
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'cache_stats': self.cache_service.get_cache_stats(),
            'plugin_stats': self.plugin_registry.get_stats(),
            'config': {
                'cache_enabled': config.cache_enabled,
                'plugins_enabled': config.plugins_enabled,
                'max_text_length': config.max_text_length,
                'ai_timeout': config.ai_timeout
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache_service.clear_cache()
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.cache_service.cleanup_expired()
