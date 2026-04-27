from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton instance - declared at module level
# Must be declared BEFORE the function that uses it
_embedder_instance = None


def get_embedder():
    global _embedder_instance

    if _embedder_instance is None:
        _embedder_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return _embedder_instance


def get_embedding_info() -> dict:
    return {
        "model": EMBEDDING_MODEL,
        "dimensions": 384,
        "device": "cpu",
        "normalized": True,
        "data_egress": False,
        "cost": 0
    }