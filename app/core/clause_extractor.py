from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from app.core.config import GROQ_API_KEY, GROQ_MODEL


# Why Pydantic model for structured output?
# We need the LLM to return consistent JSON every time.
# If we just ask "extract the termination clause" in plain text,
# sometimes it returns a paragraph, sometimes a bullet list,
# sometimes it adds extra commentary.
# with_structured_output() forces the LLM to fill in exactly
# these fields - no more, no less. Reliable every time.

class ContractClause(BaseModel):
    """Structured representation of key contract clauses."""

    termination_clause: Optional[str] = Field(
        default=None,
        description="How and when either party can terminate the contract, including notice period"
    )
    termination_page: Optional[int] = Field(
        default=None,
        description="Page number where termination clause appears"
    )

    liability_cap: Optional[str] = Field(
        default=None,
        description="Maximum liability amount or limitation of liability clause"
    )
    liability_page: Optional[int] = Field(
        default=None,
        description="Page number where liability clause appears"
    )

    governing_law: Optional[str] = Field(
        default=None,
        description="Jurisdiction and governing law for the contract"
    )
    governing_law_page: Optional[int] = Field(
        default=None,
        description="Page number where governing law appears"
    )

    payment_terms: Optional[str] = Field(
        default=None,
        description="Payment schedule, amounts, and conditions"
    )
    payment_page: Optional[int] = Field(
        default=None,
        description="Page number where payment terms appear"
    )

    renewal_terms: Optional[str] = Field(
        default=None,
        description="Auto-renewal or contract renewal conditions"
    )
    renewal_page: Optional[int] = Field(
        default=None,
        description="Page number where renewal terms appear"
    )

    risk_flags: List[str] = Field(
        default_factory=list,
        description="List of unusual, risky, or one-sided clauses that need attention"
    )

    contract_type: Optional[str] = Field(
        default=None,
        description="Type of contract: NDA, vendor agreement, employment, lease, etc."
    )

    parties: Optional[str] = Field(
        default=None,
        description="Names of the contracting parties"
    )


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise legal document analyser.
Extract key clause information from the contract text provided.

Rules:
- Extract ONLY what is explicitly stated in the contract
- If a clause is not present, return null for that field
- Always include the page number for each clause you find
- For risk_flags: identify clauses that are unusual, one-sided,
  or potentially harmful - such as auto-renewal traps, uncapped
  liability, unilateral termination rights, unlimited indemnity
- Be concise - one to two sentences per clause maximum"""),
    ("human", "Contract text:\n{contract_text}\n\nExtract the key clauses:")
])


def extract_clauses(full_text: str) -> dict:
    """
    Extracts structured clause information from contract text.

    Why do this on upload rather than on query?
    Two reasons:
    1. The analyst gets instant value the moment they upload -
       they see the summary before asking a single question
    2. It runs once at ingestion time, not on every query -
       so there is zero latency cost during the Q&A session

    This is the same pattern as a pre-processing step in an
    ML pipeline - do the expensive work once, cache the result.

    Interview answer:
    "We run structured extraction at ingestion time using
    with_structured_output() which forces the LLM to return
    a Pydantic model. This gives us consistent JSON every time
    with no post-processing needed. The analyst sees a structured
    summary the moment the upload completes."
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0  # zero temperature for deterministic extraction
    )

    # with_structured_output is the modern LangChain pattern
    # It uses function calling under the hood to force JSON output
    # matching our Pydantic schema exactly
    structured_llm = llm.with_structured_output(ContractClause)

    chain = EXTRACTION_PROMPT | structured_llm

    # Use first 6000 chars - enough for key clauses, avoids token limits
    # Why 6000? Most key clauses appear in the first 20% of a contract
    # and 6000 chars is safely within Groq's context window
    truncated_text = full_text[:6000]

    try:
        result = chain.invoke({"contract_text": truncated_text})

        # Convert Pydantic model to dict for easy JSON serialisation
        return result.model_dump()

    except Exception as e:
        print(f"  [Warning] Clause extraction failed: {e}")
        # Return empty structure - never crash the ingestion pipeline
        return ContractClause().model_dump()


def format_clause_summary(clauses: dict) -> str:
    """
    Formats extracted clauses into a readable summary.
    Used in the Streamlit UI and for logging.
    """

    lines = ["CONTRACT SUMMARY", "=" * 40]

    if clauses.get("contract_type"):
        lines.append(f"Type: {clauses['contract_type']}")

    if clauses.get("parties"):
        lines.append(f"Parties: {clauses['parties']}")

    lines.append("")

    fields = [
        ("termination_clause", "termination_page", "Termination"),
        ("liability_cap", "liability_page", "Liability Cap"),
        ("governing_law", "governing_law_page", "Governing Law"),
        ("payment_terms", "payment_page", "Payment Terms"),
        ("renewal_terms", "renewal_page", "Renewal Terms"),
    ]

    for field, page_field, label in fields:
        value = clauses.get(field)
        page = clauses.get(page_field)
        if value:
            page_ref = f" [Page {page}]" if page else ""
            lines.append(f"{label}: {value}{page_ref}")

    if clauses.get("risk_flags"):
        lines.append("")
        lines.append("RISK FLAGS:")
        for flag in clauses["risk_flags"]:
            lines.append(f"  ⚠ {flag}")

    return "\n".join(lines)