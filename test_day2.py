from app.core.ingestion import ingest_contract
from app.core.vector_store import load_vector_store

# Use the same PDF from Day 1
file_path = "data/contracts/sample_contract.pdf"

print("Running Day 2 test...\n")

# Test full ingestion pipeline
result = ingest_contract(file_path)

print("\nIngestion result:")
for key, value in result.items():
    print(f"  {key}: {value}")

# Test retrieval works
print("\nTesting retrieval...")
db = load_vector_store(result["session_id"])

# Search for something that should be in any contract
test_query = "termination"
docs = db.similarity_search(test_query, k=3)

print(f"\nTop 3 chunks for query: '{test_query}'")
print("-" * 50)
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(f"  Page: {doc.metadata['page_num']}")
    print(f"  Source: {doc.metadata['doc_name']}")
    print(f"  Preview: {doc.page_content[:200]}...")

print("\nDay 2 test PASSED" if len(docs) > 0 else "Day 2 test FAILED")