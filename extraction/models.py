# SPDX-License-Identifier: AGPL-3.0-only

"""
Pydantic models for the extraction system.

This module defines the core data structures used throughout the extraction pipeline,
providing type safety, validation, and serialization capabilities.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types for extraction."""
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FINANCIAL = "financial"
    BANK_STATEMENT = "bank_statement"
    TAX_FORM = "tax_form"
    RESEARCH = "research"
    HEALTHCARE = "healthcare"
    RESUME = "resume"
    LEGAL_PLEADING = "legal_pleading"
    PATENT = "patent"
    MEDICAL_BILL = "medical_bill"
    LAB_REPORT = "lab_report"
    INSURANCE_CLAIM = "insurance_claim"
    REAL_ESTATE = "real_estate"
    SHIPPING_MANIFEST = "shipping_manifest"
    GENERAL = "general"


class ExtractionOptions(BaseModel):
    """Configuration options for document extraction."""
    domain_override: Optional[str] = Field(None, description="Override detected document type")
    selected_fields: List[str] = Field(default_factory=list, description="Specific fields to extract")
    custom_instructions: str = Field("", description="Custom extraction instructions")
    enrich: bool = Field(False, description="Enable LLM enrichment")
    use_ocr: bool = Field(False, description="Force OCR instead of native text extraction")
    extract_tables: bool = Field(False, description="Enable table extraction")
    extract_formulas: bool = Field(False, description="Enable formula extraction")
    
    @validator('selected_fields')
    def validate_selected_fields(cls, v):
        """Ensure selected_fields is a list of strings."""
        if not isinstance(v, list):
            return []
        return [str(field) for field in v if field]


class ValidationResult(BaseModel):
    """Validation results for extracted data."""
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class ConfidenceScore(BaseModel):
    """Confidence scoring for extracted fields."""
    overall: int = Field(ge=0, le=100, description="Overall confidence score (0-100)")
    fields: Dict[str, int] = Field(default_factory=dict, description="Per-field confidence scores")


class ExtractionResult(BaseModel):
    """Complete extraction result."""
    type: str = Field(description="Detected document type")
    pages: int = Field(ge=0, description="Number of pages processed")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Raw extracted entities")
    mapped_fields: Dict[str, Any] = Field(default_factory=dict, description="Mapped domain-specific fields")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields from instructions")
    validation: ValidationResult = Field(default_factory=ValidationResult, description="Validation results")
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore, description="Confidence scores")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    formulas: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted formulas")
    provenance: Optional[Dict[str, Any]] = Field(None, description="Source page references")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create from dictionary."""
        return cls(**data)


class CacheEntry(BaseModel):
    """Cache entry for extraction results."""
    result: ExtractionResult
    timestamp: float = Field(description="Cache timestamp")
    ttl: int = Field(default=86400, description="Time to live in seconds")
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        import time
        return time.time() - self.timestamp > self.ttl


class PluginMetadata(BaseModel):
    """Metadata for extraction plugins."""
    name: str = Field(description="Plugin name")
    version: str = Field(default="1.0.0", description="Plugin version")
    description: str = Field("", description="Plugin description")
    supported_types: List[str] = Field(default_factory=list, description="Supported document types")
    priority: int = Field(default=0, description="Plugin priority (higher = more important)")


class ExtractionError(BaseModel):
    """Error information for extraction failures."""
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(default_factory=lambda: __import__('time').time(), description="Error timestamp")
