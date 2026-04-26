from app.core.ingestion import ingest_contract
from app.core.qa_chain import get_answer

# Step 1: Ingest the contract (same PDF from Day 1 and 2)
file_path = "data/contracts/sample_contract.pdf"

print("Step 1: Ingesting contract...")
result = ingest_contract(file_path)
session_id = result["session_id"]
print(f"Session ID: {session_id}")
print(f"Chunks stored: {result['total_chunks']}")

# Step 2: Ask questions
print("\nStep 2: Asking questions...\n")

questions = [
    "What is the termination clause?",
    "What is the liability cap?",
    "Which jurisdiction governs this contract?",
    "What are the payment terms?",
    "What happens if either party breaches the contract?"
]

for question in questions:
    print(f"Q: {question}")
    response = get_answer(question, session_id)
    print(f"A: {response['answer']}")
    print(f"   Confidence: {response['confidence']}")
    print(f"   Latency: {response['latency_seconds']}s")
    print(f"   Status: {response['status']}")
    if response["sources"]:
        pages = list(set(s["page"] for s in response["sources"]))
        print(f"   Pages cited: {pages}")
    print()

print("Day 3 test DONE")