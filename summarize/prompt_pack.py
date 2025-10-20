"""
Domain-specific prompt packs for Summarize tool.
"""
from typing import Dict, List


# Domain-specific prompt templates for summarization
DOMAIN_PROMPTS = {
    "legal": {
        "sections": ["Executive Summary", "Key Legal Issues", "Holdings and Decisions", "Implications", "Action Items"],
        "instructions": "Summarize legal issues, holdings, precedents, and practical implications. Focus on actionable insights.",
        "required_elements": ["issues", "holdings"]
    },
    "legal_pleading": {
        "sections": ["Case Summary", "Claims and Causes of Action", "Key Arguments", "Relief Sought", "Next Steps"],
        "instructions": "Summarize the pleading's claims, legal arguments, relief sought, and procedural posture.",
        "required_elements": ["claims", "relief"]
    },
    "finance": {
        "sections": ["Executive Summary", "Financial Highlights", "Key Metrics and Trends", "Risks and Opportunities", "Recommendations"],
        "instructions": "Summarize financial performance, key metrics, trends, risks, and strategic recommendations.",
        "required_elements": ["metrics", "trends"]
    },
    "financial": {
        "sections": ["Overview", "Key Financial Metrics", "Trends and Analysis", "Risks", "Outlook"],
        "instructions": "Summarize financial position, performance metrics, trends, risks, and forward outlook.",
        "required_elements": ["metrics", "analysis"]
    },
    "research": {
        "sections": ["Research Overview", "Key Findings", "Methodology Summary", "Implications", "Future Directions"],
        "instructions": "Summarize research question, methods, key findings, implications, and suggested future work.",
        "required_elements": ["findings", "implications"]
    },
    "healthcare": {
        "sections": ["Clinical Summary", "Key Findings", "Diagnoses and Interventions", "Outcomes", "Recommendations"],
        "instructions": "Summarize clinical context, findings, interventions, outcomes, and care recommendations.",
        "required_elements": ["findings", "outcomes"]
    },
    "medical_bill": {
        "sections": ["Summary", "Services and Charges", "Insurance Coverage", "Patient Responsibility", "Next Steps"],
        "instructions": "Summarize services, charges, insurance coverage, and patient financial responsibility.",
        "required_elements": ["charges", "responsibility"]
    },
    "lab_report": {
        "sections": ["Test Summary", "Key Results", "Abnormal Findings", "Clinical Significance", "Recommendations"],
        "instructions": "Summarize tests performed, key results, abnormal findings, and clinical recommendations.",
        "required_elements": ["results", "findings"]
    },
    "invoice": {
        "sections": ["Invoice Summary", "Key Details", "Line Items Overview", "Total and Terms"],
        "instructions": "Summarize invoice parties, key details, line items, total, and payment terms.",
        "required_elements": ["total", "terms"]
    },
    "contract": {
        "sections": ["Contract Summary", "Key Terms", "Obligations", "Important Clauses", "Action Items"],
        "instructions": "Summarize parties, subject matter, key terms, obligations, and important clauses.",
        "required_elements": ["terms", "obligations"]
    },
    "purchase_order": {
        "sections": ["PO Summary", "Items and Pricing", "Delivery Terms", "Key Details"],
        "instructions": "Summarize buyer/seller, items ordered, pricing, and delivery/payment terms.",
        "required_elements": ["items", "terms"]
    },
    "receipt": {
        "sections": ["Transaction Summary", "Items Purchased", "Payment Details"],
        "instructions": "Summarize vendor, items purchased, total, and payment method.",
        "required_elements": ["items", "total"]
    },
    "bank_statement": {
        "sections": ["Statement Summary", "Account Activity", "Key Transactions", "Balance Changes"],
        "instructions": "Summarize period, opening/closing balance, major transactions, and fees.",
        "required_elements": ["balance", "transactions"]
    },
    "tax_form": {
        "sections": ["Tax Summary", "Income and Deductions", "Tax Calculation", "Key Items"],
        "instructions": "Summarize tax year, income sources, deductions, tax owed/refund, and notable items.",
        "required_elements": ["income", "tax"]
    },
    "resume": {
        "sections": ["Candidate Summary", "Experience Highlights", "Key Skills", "Education", "Notable Achievements"],
        "instructions": "Summarize candidate profile, experience, skills, education, and achievements.",
        "required_elements": ["experience", "skills"]
    },
    "patent": {
        "sections": ["Invention Summary", "Key Claims", "Technical Approach", "Prior Art", "Commercial Potential"],
        "instructions": "Summarize invention, key claims, technical approach, prior art, and potential applications.",
        "required_elements": ["claims", "invention"]
    },
    "insurance_claim": {
        "sections": ["Claim Summary", "Incident Details", "Coverage Analysis", "Claim Amount", "Status"],
        "instructions": "Summarize claim details, incident, coverage, amount claimed, and current status.",
        "required_elements": ["incident", "amount"]
    },
    "real_estate": {
        "sections": ["Property Summary", "Key Terms", "Financial Details", "Conditions", "Important Dates"],
        "instructions": "Summarize property, parties, key terms, price, conditions, and closing details.",
        "required_elements": ["property", "terms"]
    },
    "shipping_manifest": {
        "sections": ["Shipment Summary", "Items and Quantities", "Routing", "Delivery Details"],
        "instructions": "Summarize shipper/consignee, items, routing, and delivery information.",
        "required_elements": ["items", "delivery"]
    },
    "general": {
        "sections": ["Summary", "Key Topics", "Main Points", "Recommendations"],
        "instructions": "Provide a general summary covering key topics, main points, and recommendations.",
        "required_elements": []
    }
}


def build_summarize_prompt(text: str, domain: str, detail: str, provenance: bool, custom_instructions: str = "", tables: List[Dict] = None) -> str:
    """
    Build domain-specific Summarize prompt.
    
    Args:
        text: Extracted PDF text
        domain: Document domain
        detail: "executive" or "deep"
        provenance: Include page references
        custom_instructions: User custom instructions
        tables: Optional extracted tables
    """
    domain_config = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    
    sections_str = ", ".join(domain_config["sections"])
    instructions = domain_config["instructions"]
    
    detail_guidance = ""
    if detail == "deep":
        detail_guidance = "\n- Provide comprehensive analysis with supporting details.\n- Include technical depth where appropriate."
    else:
        detail_guidance = "\n- Keep summary concise and executive-level.\n- Focus on the most critical points."
    
    provenance_guidance = ""
    if provenance:
        provenance_guidance = "\n- Include page references for each key point."
    
    table_context = ""
    if tables and len(tables) > 0:
        table_context = f"\n\nExtracted Tables:\n{str(tables[:3])}\n"
    
    custom_context = ""
    if custom_instructions:
        custom_context = f"\n\nCustom Instructions: {custom_instructions}"
    
    prompt = f"""You are an expert analyst summarizing a {domain} document.

Task: Create a structured summary with the following sections:
{sections_str}

Domain Guidance: {instructions}
{detail_guidance}{provenance_guidance}{custom_context}{table_context}

Document Text:
{text[:8000]}

Respond in JSON format:
{{
  "summary": "High-level overview (2-3 sentences)",
  "key_topics": ["topic1", "topic2", ...],
  "main_points": ["point1", "point2", ...],
  "recommendations": ["rec1", "rec2", ...]
}}

Ensure all required elements are addressed: {', '.join(domain_config['required_elements']) if domain_config['required_elements'] else 'N/A'}
"""
    
    return prompt


def get_domain_checklist(domain: str) -> List[str]:
    """Get required elements checklist for a domain."""
    domain_config = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    return domain_config.get("required_elements", [])


def get_expected_sections(domain: str) -> List[str]:
    """Get expected sections for a domain."""
    domain_config = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    return domain_config.get("sections", [])

