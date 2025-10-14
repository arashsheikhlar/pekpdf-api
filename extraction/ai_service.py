# SPDX-License-Identifier: AGPL-3.0-only

"""
AI service for document extraction.

This module handles all AI-related operations including calling different AI providers
(Ollama, OpenAI, Anthropic) and processing their responses for document extraction.
"""

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .models import ExtractionOptions, ExtractionResult, ValidationResult, ConfidenceScore
from .domain_service import DomainService
from .config import config


class AIExtractionService:
    """Service for AI-powered document extraction."""
    
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
    
    def __init__(self, domain_service: DomainService):
        """
        Initialize the AI extraction service.
        
        Args:
            domain_service: Domain service for field validation
        """
        self.domain_service = domain_service
        self.ai_config = config.get_ai_config()
    
    def extract_document(self, text: str, pages_text: List[str], options: ExtractionOptions) -> ExtractionResult:
        """
        Extract document information using AI.
        
        Args:
            text: Combined text from all pages
            pages_text: List of text for each page
            options: Extraction options
            
        Returns:
            Extraction result with extracted data
        """
        # Smart chunking to keep guidance intact but cap tokens
        max_chars = config.max_text_length
        if len(text) > max_chars:
            head = text[:10000]
            middle = text[len(text)//2 - 2500: len(text)//2 + 2500]
            tail = text[-5000:]
            doc_snippet = f"{head}\n\n[MIDDLE]\n\n{middle}\n\n[END]\n\n{tail}"
        else:
            doc_snippet = text
        
        system_prompt = (
            "You are an expert document information extractor. Return ONLY valid JSON. "
            "Extract exact values from the text without paraphrasing, keep original formatting of numbers and dates where possible. "
            "Use null for missing fields."
        )
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(doc_snippet, options)
        
        # Call AI service with timeout
        ai_response = self._call_ai_with_timeout(prompt, system_prompt)
        
        # Parse AI response
        try:
            data = self.parse_json_safely(ai_response)
        except Exception:
            # Fallback to empty result if parsing fails
            data = {}
        
        # Build extraction result
        return self._build_extraction_result(data, pages_text, options)
    
    def enrich_extraction(self, dtype: str, mapped: Dict[str, Any], text: str, pages_text: List[str], options: ExtractionOptions) -> Dict[str, Any]:
        """
        Enrich extraction with additional LLM processing.
        
        Args:
            dtype: Document type
            mapped: Already extracted fields
            text: Full document text
            pages_text: List of page text
            options: Extraction options
            
        Returns:
            Dictionary with enriched mapped_fields
        """
        try:
            if not isinstance(mapped, dict) or not text:
                return {"mapped_fields": mapped}
            
            domain = (dtype or 'general').lower()
            
            # Build RAG snippets based on domain
            rag_snippets = self._build_rag_snippets(domain, mapped, pages_text)
            rag_block = ("\nRAG_SNIPPETS:\n" + "\n\n".join(rag_snippets)) if rag_snippets else ""
            
            prompt = (
                f"SYSTEM: You are a domain-specific extraction validator and enricher for {domain} documents.\n"
                "Return ONLY a JSON object with normalized structures.\n\n"
                "INPUT (truncated to 8000 chars):\n" + text[:8000] + "\n\n"
                + rag_block + "\n\n"
                "TASK:\n"
                "- Normalize and enrich extracted fields if possible.\n"
                "- For healthcare: infer diagnoses (array of strings), procedures (array), and normalize labs as {name,value,unit,flag}.\n"
                "- For contracts: extract obligations (array), termination_conditions (array).\n"
                "- For research: extract key_findings (array) and primary_outcomes (array).\n\n"
                "OUTPUT (strict JSON): {\n"
                "  \"diagnoses\": [], \"procedures\": [], \"labs_normalized\": [], \"obligations\": [], \"termination_conditions\": [], \"key_findings\": [], \"primary_outcomes\": []\n"
                "}"
            )
            
            ai_response = self._call_ai_with_timeout(prompt)
            data = self.parse_json_safely(ai_response)
            
            if not isinstance(data, dict):
                return {"mapped_fields": mapped}
            
            # Merge enrichment data
            out = dict(mapped)
            enrichment = {}
            for k in ("diagnoses", "procedures", "labs_normalized", "obligations", "termination_conditions", "key_findings", "primary_outcomes"):
                if k in data and data.get(k) is not None:
                    enrichment[k] = data.get(k)
            
            if enrichment:
                out["enriched"] = True
                out["enrichment"] = enrichment
            
            return {"mapped_fields": out}
            
        except Exception:
            return {"mapped_fields": mapped}
    
    def _build_extraction_prompt(self, doc_snippet: str, options: ExtractionOptions) -> str:
        """Build the extraction prompt based on options."""
        domain_norm = self.domain_service.normalize_domain(options.domain_override)
        allowed_fields = self.domain_service.get_domain_schema(domain_norm)
        selected = self.domain_service.validate_selected_fields(domain_norm, options.selected_fields)
        
        if domain_norm and domain_norm in self.domain_service.DOMAIN_FIELDS:
            fields_for_prompt = selected if selected else allowed_fields
            prompt = (
                f"Document domain: {domain_norm}. Extract ONLY the following fields under 'mapped_fields': {json.dumps(fields_for_prompt)}\n"
                "Return JSON with keys: {\n"
                "  \"type\": string (exactly the provided domain),\n"
                "  \"mapped_fields\": object (only the requested fields),\n"
                "  \"entities\": { \"emails\": [], \"phones\": [], \"amounts\": [], \"dates\": [] },\n"
                "  \"custom_fields\": object | null\n"
                "}\n\n"
                "Rules: Use exact values from text; preserve number/date formatting; missing => null (or [] for arrays).\n"
                "Do not include extraneous keys. Return ONLY JSON.\n\n"
                f"Text (truncated):\n{doc_snippet}\n"
            )
        else:
            # Compact auto-detect fallback
            compact_schema = {k: self.domain_service.DOMAIN_FIELDS[k][:8] for k in ("invoice", "contract", "financial", "research", "healthcare", "general") if k in self.domain_service.DOMAIN_FIELDS}
            prompt = (
                "Detect document type as one of: invoice, contract, financial, research, healthcare, general.\n"
                "Then extract fields (limited set) as strict JSON with keys: {\n"
                "  \"type\": string,\n"
                "  \"mapped_fields\": object,\n"
                "  \"entities\": { \"emails\": [], \"phones\": [], \"amounts\": [], \"dates\": [] },\n"
                "  \"custom_fields\": object | null\n"
                "}\n\n"
                f"Text (truncated):\n{doc_snippet}\n\n"
            )
        
        # Add custom instructions if provided
        if options.custom_instructions:
            prompt += (
                "\nIMPORTANT: The user provided CUSTOM INSTRUCTIONS below.\n"
                "DECISION LOGIC:\n"
                "- If the instruction relates to existing document fields (e.g., 'extract dates in ISO format', 'focus on medication dosages'), modify the relevant mapped_fields accordingly.\n"
                "- If the instruction asks for different concepts/data types (e.g., 'extract keywords', 'find risk factors', 'identify action items'), create new fields in 'custom_fields'.\n"
                "- You can do BOTH: modify existing fields AND add custom fields if the instruction covers both.\n"
                "- Use descriptive snake_case keys for custom_fields (e.g., 'keywords', 'risk_factors', 'action_items').\n"
                "- If nothing relevant is found, set 'custom_fields' to null.\n\n"
                f"CUSTOM INSTRUCTIONS:\n{options.custom_instructions}\n"
            )
        
        return prompt
    
    def _build_rag_snippets(self, domain: str, mapped: Dict[str, Any], pages_text: List[str]) -> List[str]:
        """Build RAG snippets for enrichment."""
        rag_snippets = []
        try:
            # Build inverted index for page references
            inv_index = {}
            for i, txt in enumerate(pages_text or []):
                if not txt:
                    continue
                words = txt.lower().split()
                for word in words:
                    if len(word) >= 4:
                        if word not in inv_index:
                            inv_index[word] = []
                        inv_index[word].append(i + 1)
            
            # Extract tokens from mapped fields for domain-specific RAG
            tokens = []
            if domain == 'healthcare':
                for key in ['diagnosis_text', 'plan']:
                    if isinstance(mapped.get(key), str):
                        tokens.extend(mapped[key].split())
                for med in (mapped.get('medications') or []):
                    if isinstance(med, dict) and med.get('name'):
                        tokens.extend(med['name'].split())
            elif domain == 'contract':
                for key in ['party_a', 'party_b', 'governing_law', 'term']:
                    if isinstance(mapped.get(key), str):
                        tokens.extend(mapped[key].split())
            elif domain == 'research':
                for key in ['abstract', 'methodology', 'results', 'conclusions']:
                    if isinstance(mapped.get(key), str):
                        tokens.extend(mapped[key].split())
            
            # Find relevant pages
            relevant_pages = set()
            for token in tokens:
                if token.lower() in inv_index:
                    relevant_pages.update(inv_index[token.lower()])
            
            # Build snippets from relevant pages
            for page_num in sorted(relevant_pages)[:5]:  # Limit to 5 pages
                if 1 <= page_num <= len(pages_text):
                    snippet = pages_text[page_num - 1][:500]  # Limit snippet length
                    rag_snippets.append(f"Page {page_num}: {snippet}")
        
        except Exception:
            pass
        
        return rag_snippets
    
    def _call_ai_with_timeout(self, prompt: str, system_prompt: str = "") -> str:
        """Call AI service with timeout handling."""
        result_holder = {"text": None, "error": None}
        
        def _runner():
            try:
                result_holder["text"] = self._call_ai_service(prompt, system_prompt)
            except Exception as e:
                result_holder["error"] = str(e)
        
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=config.ai_timeout)
        
        if thread.is_alive():
            result_holder["text"] = None
            result_holder["error"] = "timeout"
        
        if result_holder["error"] == "timeout":
            return ""
        
        return result_holder["text"] or ""
    
    def _call_ai_service(self, prompt: str, system_prompt: str = "") -> str:
        """Call the configured AI service."""
        service = self.ai_config["service"]
        
        if service == "openai":
            return self._call_openai(prompt, system_prompt)
        elif service == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        else:
            return self._call_ollama(prompt, system_prompt)
    
    def _call_openai(self, prompt: str, system_prompt: str = "") -> str:
        """Call OpenAI API."""
        openai_config = self.ai_config["openai"]
        if not openai_config["api_key"]:
            return "Error: OpenAI API key not configured."
        
        try:
            import openai
            openai.api_key = openai_config["api_key"]
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model=openai_config["model"],
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def _call_anthropic(self, prompt: str, system_prompt: str = "") -> str:
        """Call Anthropic API."""
        anthropic_config = self.ai_config["anthropic"]
        if not anthropic_config["api_key"]:
            return "Error: Anthropic API key not configured."
        
        model = anthropic_config["model"]
        if model not in self.VALID_ANTHROPIC_MODELS:
            return f"Error: Invalid Anthropic model '{model}'."
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "user", "content": f"System: {system_prompt}\n\nUser: {prompt}"})
            else:
                messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": anthropic_config["api_key"],
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                return f"Error: Anthropic API returned status {response.status_code}"
        
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"
    
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama API."""
        ollama_config = self.ai_config["ollama"]
        
        try:
            url = f"{ollama_config['base_url']}/api/generate"
            payload = {
                "model": ollama_config["model"],
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "num_ctx": ollama_config["num_ctx"],
                    "num_predict": ollama_config["num_predict"],
                    "repeat_penalty": 1.1,
                    "seed": -1,
                    "stop": ["```", "\n\n\n"]
                }
            }
            
            response = requests.post(url, json=payload, timeout=config.ai_timeout)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response from AI model')
            else:
                return f"Error calling Ollama: {response.status_code}"
        
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from AI response with error handling.
        
        Args:
            text: Raw AI response text
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        if not isinstance(text, str):
            raise ValueError("Response is not a string")
        
        cleaned = text.strip()
        
        # Remove common code fences
        if cleaned.startswith("```"):
            cleaned = cleaned.strip('`')
            cleaned = cleaned.replace('json', '', 1).strip()
        
        # Try direct JSON parse first
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
    
    def _build_extraction_result(self, data: Dict[str, Any], pages_text: List[str], options: ExtractionOptions) -> ExtractionResult:
        """Build ExtractionResult from AI response data."""
        dtype = data.get('type', 'general')
        entities = data.get('entities', {})
        mapped = data.get('mapped_fields', {})
        custom_fields = data.get('custom_fields')
        
        # Clean mapped fields
        mapped_clean = {}
        domain_norm = self.domain_service.normalize_domain(options.domain_override)
        allowed_fields = self.domain_service.get_domain_schema(domain_norm)
        
        for k, v in mapped.items():
            if k in allowed_fields:
                mapped_clean[k] = v
        
        # Build confidence score
        fields_total = len(allowed_fields)
        fields_present = sum(1 for v in mapped_clean.values() if v)
        overall = int((fields_present / fields_total) * 100) if fields_total else 60
        confidence = ConfidenceScore(
            overall=overall,
            fields={k: (85 if mapped_clean.get(k) else 0) for k in allowed_fields}
        )
        
        return ExtractionResult(
            type=dtype,
            pages=len(pages_text),
            entities=entities,
            mapped_fields=mapped_clean,
            custom_fields=custom_fields,
            validation=ValidationResult(),
            confidence=confidence,
            tables=[],
            formulas=[],
            provenance=None
        )
