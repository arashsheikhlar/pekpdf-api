"""
Input validation schemas using Marshmallow for API endpoints.
"""
from marshmallow import Schema, fields, validate, ValidationError


class ChatRequestSchema(Schema):
    """Validation schema for AI chat requests."""
    question = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=2000),
        error_messages={
            'required': 'Question field is required',
            'invalid': 'Question must be a string'
        }
    )


class ExtractRequestSchema(Schema):
    """Validation schema for data extraction requests."""
    domain = fields.Str(
        required=False,
        validate=validate.Length(min=1, max=50),
        allow_none=True,
        error_messages={'invalid': 'Domain must be a valid string'}
    )
    selected_fields = fields.List(
        fields.Str(),
        required=False,
        allow_none=True,
        validate=validate.Length(max=100),  # Max 100 fields
        error_messages={'invalid': 'Selected fields must be a list of strings'}
    )
    custom_instructions = fields.Str(
        required=False,
        allow_none=True,
        validate=validate.Length(max=1000),
        error_messages={'invalid': 'Custom instructions must be a string'}
    )
    use_ocr = fields.Bool(required=False, load_default=False)
    export_format = fields.Str(
        required=False,
        validate=validate.OneOf(['json', 'csv', 'pdf']),
        load_default='json',
        error_messages={'invalid': 'Export format must be json, csv, or pdf'}
    )


class PageRangeSchema(Schema):
    """Validation schema for page number ranges."""
    pages = fields.Str(
        required=False,
        validate=validate.Length(max=100),
        error_messages={'invalid': 'Page range must be a string'}
    )
    delete = fields.Str(
        required=False,
        validate=validate.Length(max=100),
        error_messages={'invalid': 'Delete list must be a string'}
    )
    order = fields.Str(
        required=False,
        validate=validate.Length(max=100),
        error_messages={'invalid': 'Order list must be a string'}
    )


class FileUploadSchema(Schema):
    """Validation schema for file uploads."""
    filename = fields.Str(
        required=False,
        validate=validate.Length(max=255),
        error_messages={'invalid': 'Filename must be a valid string'}
    )
    content_length = fields.Int(
        required=False,
        validate=validate.Range(min=0, max=100 * 1024 * 1024),  # 100MB max
        error_messages={'invalid': 'File size must be valid'}
    )


class ExplainRequestSchema(Schema):
    """Validation schema for AI explain requests."""
    domain = fields.Str(
        required=False,
        allow_none=True,
        validate=validate.Length(min=1, max=50),
        load_default='general',
        error_messages={'invalid': 'Domain must be a valid string'}
    )
    detail = fields.Str(
        required=False,
        validate=validate.OneOf(['basic', 'advanced', 'executive']),
        load_default='basic',
        error_messages={'invalid': 'Detail must be basic, advanced, or executive'}
    )


class SummarizeRequestSchema(Schema):
    """Validation schema for AI summarize requests."""
    template = fields.Str(
        required=False,
        validate=validate.Length(max=50),
        error_messages={'invalid': 'Template must be a valid string'}
    )
    format = fields.Str(
        required=False,
        validate=validate.OneOf(['report', 'brief', 'minutes']),
        error_messages={'invalid': 'Format must be report, brief, or minutes'}
    )

