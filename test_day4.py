from app.core.ingestion import ingest_contract
from app.core.qa_chain import get_answer

file_path = "data/contracts/vendor.pdf"

print("Running Day 4 test - tables, images, structured clauses\n")

result = ingest_contract(file_path)

print("\n" + "="*55)
print("STRUCTURED CLAUSE SUMMARY:")
print("="*55)
print(result["clause_summary"])

print("\n" + "="*55)
print("CHUNK BREAKDOWN:")
print("="*55)
print(f"Text chunks  : {result['text_chunks']}")
print(f"Table chunks : {result['table_chunks']}")
print(f"Image chunks : {result['image_chunks']}")
print(f"Total        : {result['total_chunks']}")

print("\n" + "="*55)
print("Q&A TEST:")
print("="*55)

questions = [
    "What is the termination clause?",
    "What are the payment terms?",
    "What is the governing law?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = get_answer(q, result["session_id"])
    print(f"A: {response['answer'][:300]}")
    print(f"   Status: {response['status']} | Confidence: {response['confidence']}")

print("\nDay 4 DONE")