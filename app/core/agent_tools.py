from langchain.tools import tool
from app.core.vector_store import load_vector_store, get_retriever
from langchain_core.documents import Document
from typing import List


# CORRECT AGENT PATTERN:
# Tools fetch and return RAW TEXT only
# The agent's LLM generates the final answer from tool output
# Tools never call the LLM internally
# This avoids nested LLM calls and token limit issues


def format_docs_simple(docs: List[Document]) -> str:
    """
    Formats retrieved chunks into plain text with page markers.
    No LLM involved — pure text formatting.
    """
    if not docs:
        return "No relevant information found in the contract."

    parts = []
    for doc in docs:
        page = doc.metadata.get("page_num", "?")
        content_type = doc.metadata.get("content_type", "text")
        parts.append(
            f"[Page {page}] ({content_type})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def make_tools(session_id: str):
    """
    Creates 3 tools that ONLY retrieve data — no LLM calls inside.
    The agent's own LLM generates the final answer from returned text.
    """

    @tool
    def retrieve_and_answer(question: str) -> str:
        """
        Retrieves relevant contract clauses for a specific question.
        Returns the raw contract text with page numbers.
        Use for specific questions about clauses, terms, parties,
        dates, payments, obligations, or any factual contract question.
        """
        try:
            retriever = get_retriever(session_id)
            docs = retriever.invoke(question)

            if not docs:
                return "No relevant clauses found for this question."

            return format_docs_simple(docs)

        except Exception as e:
            return f"Retrieval error: {str(e)}"

    @tool
    def summarise_contract(request: str) -> str:
        """
        Retrieves key sections from across the entire contract.
        Returns raw contract text covering main terms and conditions.
        Use when the user wants a summary or overview of the contract.
        """
        try:
            db = load_vector_store(session_id)

            if db is None:
                return "No contract loaded for this session."

            # Search for broad coverage of contract content
            docs = db.similarity_search(
                "parties agreement terms conditions obligations",
                k=3
            )

            if not docs:
                return "Could not retrieve contract content."

            return format_docs_simple(docs)

        except Exception as e:
            return f"Summary retrieval error: {str(e)}"

    @tool
    def flag_contract_risks(concern: str) -> str:
        """
        Retrieves clauses related to liability, indemnity, termination,
        and other risk areas from the contract.
        Returns raw contract text from risk-related sections.
        Use when the user asks about risks, red flags, or unusual terms.
        """
        try:
            db = load_vector_store(session_id)

            if db is None:
                return "No contract loaded for this session."

            # Search for risk-related content
            docs = db.similarity_search(
                "liability indemnity termination penalty damages",
                k=3
            )

            if not docs:
                return "No risk-related clauses found."

            return format_docs_simple(docs)

        except Exception as e:
            return f"Risk retrieval error: {str(e)}"

    return [retrieve_and_answer, summarise_contract, flag_contract_risks]