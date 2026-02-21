import os
import re
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI


qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

PRODUCT_PATTERN = re.compile(r"\bPROD\d+\b", re.IGNORECASE)


def retrieve_products(query: str, limit: int = 3) -> Dict[str, Any]:

    # ==========================================================
    # 1️⃣ Deterministic Product ID Lookup
    # ==========================================================
    match = PRODUCT_PATTERN.search(query)

    if match:
        product_id = match.group().upper()

        results, _ = qdrant.scroll(
            collection_name="products_collection",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="product_id",
                        match=MatchValue(value=product_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if results:
            return {
                "retrieval_type": "deterministic",
                "confidence": 1.0,
                "max_similarity": 1.0,
                "results": [point.payload for point in results]
            }

        return {
            "retrieval_type": "deterministic",
            "confidence": 0.0,
            "error": f"Product {product_id} not found"
        }

    # ==========================================================
    # 2️⃣ Semantic Fallback
    # ==========================================================

    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = qdrant.query_points(
        collection_name="products_collection",
        query=embedding,
        limit=limit,
        with_payload=True
    )

    if not results.points:
        return {
            "retrieval_type": "semantic",
            "confidence": 0.0,
            "results": []
        }

    retrieved_results: List[Dict[str, Any]] = []
    max_similarity = 0.0

    for hit in results.points:
        max_similarity = max(max_similarity, hit.score)

        retrieved_results.append({
            "product_id": hit.payload.get("product_id"),
            "name": hit.payload.get("name"),
            "price": hit.payload.get("price"),
            "similarity_score": round(hit.score, 4),
            "payload": hit.payload
        })

    return {
        "retrieval_type": "semantic",
        "confidence": round(max_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "results": retrieved_results
    }
