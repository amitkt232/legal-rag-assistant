import pdfplumber
import fitz  # PyMuPDF - already installed
from typing import List, Dict


def extract_tables_from_pdf(file_path: str) -> List[Dict]:
    """
    Extracts tables from a PDF and converts them to clean text.

    Why pdfplumber for tables?
    PyMuPDF reads PDF text but flattens table structure.
    "Payment | Month 1 | Month 2" becomes "Payment Month 1 Month 2"
    - completely loses which value belongs to which column.

    pdfplumber understands PDF table geometry - it reads the actual
    cell boundaries and reconstructs rows and columns properly.

    Why does this matter for legal contracts?
    Payment schedules, penalty tables, fee structures are all in tables.
    A flattened table gives wrong answers. A structured table gives
    correct answers with proper row/column context.

    Interview answer for "how did you handle table data":
    "We used pdfplumber which understands PDF table geometry and
    reconstructs rows and columns. We then converted each table to
    a markdown format before chunking - this preserves structure
    in a way that both the embedder and LLM can understand."

    Returns list of dicts with page_num, table_text, table_index
    """

    tables_found = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):

                # extract_tables() returns list of lists
                # Each table is a list of rows
                # Each row is a list of cell values
                page_tables = page.extract_tables()

                if not page_tables:
                    continue

                for table_idx, table in enumerate(page_tables):
                    if not table:
                        continue

                    # Convert table to markdown format
                    # Why markdown? LLMs are trained on markdown tables
                    # and understand them much better than plain text
                    table_text = convert_table_to_markdown(
                        table,
                        page_num,
                        table_idx
                    )

                    if table_text:
                        tables_found.append({
                            "page_num": page_num,
                            "table_index": table_idx,
                            "table_text": table_text,
                            "row_count": len(table),
                            "col_count": len(table[0]) if table else 0
                        })

    except Exception as e:
        # Do not crash the whole pipeline if table extraction fails
        # Log and continue - text extraction already handled the page
        print(f"  [Warning] Table extraction error: {e}")
        return []

    return tables_found


def convert_table_to_markdown(
    table: List[List],
    page_num: int,
    table_idx: int
) -> str:
    """
    Converts a raw table (list of lists) to markdown format.

    Example input:
    [["Payment", "Amount", "Due Date"],
     ["Milestone 1", "$50,000", "Month 3"],
     ["Milestone 2", "$50,000", "Month 6"]]

    Example output:
    [Table 1, Page 4]
    | Payment | Amount | Due Date |
    |---------|--------|----------|
    | Milestone 1 | $50,000 | Month 3 |
    | Milestone 2 | $50,000 | Month 6 |

    The LLM reads this naturally and can answer:
    "When is Milestone 2 payment due?" -> "Month 6 [Page 4, Table 1]"
    """

    if not table or not table[0]:
        return ""

    lines = [f"\n[Table {table_idx + 1}, Page {page_num}]"]

    # Clean None values - pdfplumber returns None for empty cells
    cleaned_table = []
    for row in table:
        cleaned_row = [
            str(cell).strip() if cell is not None else ""
            for cell in row
        ]
        cleaned_table.append(cleaned_row)

    # First row as header
    header = cleaned_table[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # Remaining rows as data
    for row in cleaned_table[1:]:
        # Pad row if it has fewer columns than header
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def get_table_summary(tables: List[Dict]) -> str:
    """Summary for logging and debugging."""
    if not tables:
        return "No tables found"
    return (
        f"Found {len(tables)} tables across "
        f"{len(set(t['page_num'] for t in tables))} pages"
    )