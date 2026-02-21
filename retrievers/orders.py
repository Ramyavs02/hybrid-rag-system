import os
import re
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

# -----------------------------
# Clients
# -----------------------------
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -----------------------------
# Regex Pattern for Order ID
# -----------------------------
ORDER_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)


# -----------------------------
# Retrieve Orders (Hybrid)
# -----------------------------
def retrieve_orders(query: str, user_id: str = None, limit: int = 3) -> Dict[str, Any]:

    # ==========================================================
    # 1️⃣ Deterministic Order ID Lookup
    # ==========================================================
    match = ORDER_PATTERN.search(query)

    if match:
        order_id = match.group().upper()

        must_conditions = [
            FieldCondition(
                key="order_id",
                match=MatchValue(value=order_id)
            )
        ]

        # 🔐 Optional: Add user-level security filter
        if user_id:
            must_conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            )

        results, _ = qdrant.scroll(
            collection_name="orders_collection",
            scroll_filter=Filter(must=must_conditions),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if results:
            order = results[0]

            return {
                "retrieval_type": "deterministic",
                "confidence": 1.0,  # deterministic = full confidence
                "max_similarity": 1.0,
                "results": [
                    {
                        "order_id": order.payload.get("order_id"),
                        "status": order.payload.get("status"),
                        "total_amount": order.payload.get("total_amount"),
                        "created_at": order.payload.get("created_at"),
                        "payload": order.payload
                    }
                ]
            }

        return {
            "retrieval_type": "deterministic",
            "confidence": 0.0,
            "error": f"Order {order_id} not found"
        }

    # ==========================================================
    # 2️⃣ Semantic Search Fallback
    # ==========================================================

    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    search_filter = None

    # 🔐 Optional: restrict by user_id if provided
    if user_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

    results = qdrant.search(
        collection_name="orders",
        query_vector=embedding,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )

    if not results:
        return {
            "retrieval_type": "semantic",
            "confidence": 0.0,
            "error": "No relevant orders found"
        }

    retrieved_results: List[Dict[str, Any]] = []
    max_similarity = 0.0

    for hit in results:
        max_similarity = max(max_similarity, hit.score)

        retrieved_results.append({
            "order_id": hit.payload.get("order_id"),
            "status": hit.payload.get("status"),
            "similarity_score": round(hit.score, 4),
            "payload": hit.payload
        })

    return {
        "retrieval_type": "semantic",
        "confidence": round(max_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "results": retrieved_results
    }
