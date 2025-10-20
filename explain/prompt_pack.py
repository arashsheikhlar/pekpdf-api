"""
Domain-specific prompt packs for Explain tool.
"""
from typing import Dict, List


# Domain-specific prompt templates
DOMAIN_PROMPTS = {
    "legal": {
        "outline": ["Jurisdiction", "Parties", "Issues", "Holdings", "Precedents Cited", "Legal Reasoning", "Implications"],
        "instructions": "Focus on legal issues, holdings, precedents, and jurisdictional context. Identify parties, key legal questions, and the court's reasoning.",
        "required_elements": ["jurisdiction", "parties", "issues"]
    },
    "legal_pleading": {
        "outline": ["Court and Jurisdiction", "Parties", "Causes of Action", "Legal Issues", "Relief Sought", "Key Arguments"],
        "instructions": "Identify the court, parties, causes of action, legal issues, relief sought, and key legal arguments.",
        "required_elements": ["court", "parties", "causes"]
    },
    "finance": {
        "outline": ["Period", "Key Financial Metrics", "Revenue Drivers", "Cost Structure", "Risks", "Trends"],
        "instructions": "Focus on financial period, KPIs (revenue, profit, margins), drivers, cost breakdown, and financial risks.",
        "required_elements": ["period", "metrics"]
    },
    "financial": {
        "outline": ["Period", "Key Metrics", "Assets and Liabilities", "Cash Flow", "Risks", "Outlook"],
        "instructions": "Explain financial position, key metrics, asset/liability breakdown, cash flow, and risks.",
        "required_elements": ["period", "metrics"]
    },
    "research": {
        "outline": ["Background", "Research Question", "Methods", "Results", "Limitations", "Implications"],
        "instructions": "Describe research background, question, methodology, key findings, limitations, and implications for the field.",
        "required_elements": ["methods", "results"]
    },
    "healthcare": {
        "outline": ["Patient Context", "Diagnosis", "Interventions", "Outcomes", "Safety Considerations"],
        "instructions": "Explain patient context, diagnosis, medical interventions, clinical outcomes, and safety considerations.",
        "required_elements": ["diagnosis", "interventions"]
    },
    "medical_bill": {
        "outline": ["Patient", "Provider", "Services", "Charges", "Insurance", "Payment Due"],
        "instructions": "Explain patient, provider, services rendered, charges, insurance coverage, and amounts due.",
        "required_elements": ["patient", "provider", "services"]
    },
    "lab_report": {
        "outline": ["Patient", "Tests Performed", "Results", "Reference Ranges", "Interpretations", "Recommendations"],
        "instructions": "Explain tests performed, results with reference ranges, interpretations, and clinical recommendations.",
        "required_elements": ["tests", "results"]
    },
    "invoice": {
        "outline": ["Parties", "Invoice Details", "Line Items", "Totals", "Payment Terms"],
        "instructions": "Explain vendor, customer, invoice number and date, line items, totals, and payment terms.",
        "required_elements": ["vendor", "total"]
    },
    "contract": {
        "outline": ["Parties", "Subject Matter", "Key Terms", "Obligations", "Termination", "Governing Law"],
        "instructions": "Identify parties, subject matter, key contractual terms, obligations, termination clauses, and governing law.",
        "required_elements": ["parties", "terms"]
    },
    "purchase_order": {
        "outline": ["Buyer and Seller", "PO Details", "Items Ordered", "Pricing", "Delivery Terms"],
        "instructions": "Explain buyer, seller, PO number/date, items ordered, pricing, and delivery terms.",
        "required_elements": ["buyer", "items"]
    },
    "receipt": {
        "outline": ["Vendor", "Transaction Details", "Items Purchased", "Total", "Payment Method"],
        "instructions": "Explain vendor, transaction date/ID, items, total, and payment method.",
        "required_elements": ["vendor", "total"]
    },
    "bank_statement": {
        "outline": ["Account Holder", "Period", "Opening/Closing Balance", "Transactions", "Fees"],
        "instructions": "Explain account holder, statement period, opening/closing balance, transaction summary, and fees.",
        "required_elements": ["account", "period", "balance"]
    },
    "tax_form": {
        "outline": ["Taxpayer", "Tax Year", "Income", "Deductions", "Tax Owed/Refund"],
        "instructions": "Explain taxpayer, tax year, income sources, deductions, and tax owed or refund amount.",
        "required_elements": ["taxpayer", "year"]
    },
    "resume": {
        "outline": ["Candidate", "Experience", "Education", "Skills", "Achievements"],
        "instructions": "Summarize candidate's background, work experience, education, key skills, and notable achievements.",
        "required_elements": ["candidate", "experience"]
    },
    "patent": {
        "outline": ["Invention Title", "Inventors", "Abstract", "Claims", "Prior Art", "Technical Details"],
        "instructions": "Explain invention title, inventors, abstract, key claims, prior art references, and technical details.",
        "required_elements": ["title", "claims"]
    },
    "insurance_claim": {
        "outline": ["Policyholder", "Claim Details", "Incident", "Coverage", "Amount Claimed"],
        "instructions": "Explain policyholder, claim number/date, incident description, coverage, and amount claimed.",
        "required_elements": ["policyholder", "incident"]
    },
    "real_estate": {
        "outline": ["Property", "Parties", "Terms", "Price", "Conditions", "Closing"],
        "instructions": "Explain property details, parties, key terms, price, conditions, and closing information.",
        "required_elements": ["property", "parties"]
    },
    "shipping_manifest": {
        "outline": ["Shipper and Consignee", "Shipment Details", "Items", "Routing", "Delivery"],
        "instructions": "Explain shipper, consignee, shipment ID/date, items, routing, and delivery details.",
        "required_elements": ["shipper", "items"]
    },
    "general": {
        "outline": ["Overview", "Key Concepts", "Main Points", "Context", "Implications"],
        "instructions": "Provide a general explanation of the document covering overview, key concepts, main points, context, and implications.",
        "required_elements": []
    }
}


def build_explain_prompt(text: str, domain: str, detail: str, custom_instructions: str = "", tables: List[Dict] = None) -> str:
    """
    Build domain-specific Explain prompt.
    
    Args:
        text: Extracted PDF text
        domain: Document domain
        detail: "basic" or "advanced"
        custom_instructions: User custom instructions
        tables: Optional extracted tables
    """
    domain_config = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    
    outline_str = ", ".join(domain_config["outline"])
    instructions = domain_config["instructions"]
    
    detail_guidance = ""
    if detail == "advanced":
        detail_guidance = "\n- Provide technical depth and domain-specific terminology.\n- Include nuanced analysis and implications."
    else:
        detail_guidance = "\n- Keep explanations clear and accessible.\n- Focus on the most important points."
    
    table_context = ""
    if tables and len(tables) > 0:
        table_context = f"\n\nExtracted Tables:\n{str(tables[:3])}\n"  # Limit to first 3 tables
    
    custom_context = ""
    if custom_instructions:
        custom_context = f"\n\nCustom Instructions: {custom_instructions}"
    
    prompt = f"""You are an expert analyst explaining a {domain} document.

Task: Explain this document with the following structure:
{outline_str}

Domain Guidance: {instructions}
{detail_guidance}{custom_context}{table_context}

Document Text:
{text[:8000]}

Respond in JSON format:
{{
  "summary": "Brief 2-3 sentence overview",
  "key_concepts": ["concept1", "concept2", ...],
  "explanations": {{
    "{domain_config['outline'][0]}": "explanation...",
    ...
  }},
  "context": "Background and context for understanding this document",
  "definitions": {{"term1": "definition", "term2": "definition"}}
}}

Ensure all required elements are addressed: {', '.join(domain_config['required_elements']) if domain_config['required_elements'] else 'N/A'}
"""
    
    return prompt


def get_domain_checklist(domain: str) -> List[str]:
    """Get required elements checklist for a domain."""
    domain_config = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    return domain_config.get("required_elements", [])

