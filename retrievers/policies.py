import os
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from openai import OpenAI

# -----------------------------
# Clients (CLOUD CONFIG)
# -----------------------------
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -----------------------------
# Retrieve Policies (Semantic Only - Cloud Safe)
# -----------------------------
def retrieve_policies(query: str, limit: int = 3) -> Dict[str, Any]:

    # Create embedding
    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Semantic search in Qdrant
    results = qdrant.search(
        collection_name="policies_collection",
        query_vector=embedding,
        limit=limit,
        with_payload=True
    )

    if not results:
        return {
            "retrieval_type": "semantic",
            "confidence": 0.0,
            "max_similarity": 0.0,
            "results": []
        }

    retrieved_results: List[Dict[str, Any]] = []
    max_similarity = 0.0

    for hit in results:
        max_similarity = max(max_similarity, hit.score)

        retrieved_results.append({
            "similarity_score": round(hit.score, 4),
            "payload": hit.payload
        })

    return {
        "retrieval_type": "semantic",
        "confidence": round(max_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "results": retrieved_results
    }
