from app.core.pdf_extractor import (
    extract_text_from_pdf,
    get_full_text,
    get_extraction_summary
)

# Download any sample contract PDF from Google
# Search: "sample NDA PDF download" or "sample vendor agreement PDF"
# Save it to: data/contracts/sample_contract.pdf

file_path = "data/contracts/sample_contract.pdf"

print("Running Day 1 test...\n")

result = extract_text_from_pdf(file_path)

print(get_extraction_summary(result))
print(f"\nFirst page preview:")
print("-" * 50)
print(result["pages"][0]["text"][:500])
print("-" * 50)
print(f"\nTotal characters extracted: {sum(len(p['text']) for p in result['pages'])}")
print("\nDay 1 test PASSED" if result["total_pages"] > 0 else "Day 1 test FAILED")
