import os
import re
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
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

POLICY_PATTERN = re.compile(r"\bPOL\d+\b", re.IGNORECASE)

POLICY_KEYWORDS = [
    "refund",
    "return",
    "warranty",
    "shipping",
    "cancellation",
    "privacy",
    "payment",
    "exchange"
]


def retrieve_policies(query: str, limit: int = 3) -> Dict[str, Any]:

    query_lower = query.lower()

    # ==========================================================
    # 1️⃣ Deterministic Policy ID Lookup
    # ==========================================================
    match = POLICY_PATTERN.search(query)

    if match:
        policy_id = match.group().upper()

        results, _ = qdrant.scroll(
            collection_name="policies_collection",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="policy_id",
                        match=MatchValue(value=policy_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if results:
            return {
                "retrieval_type": "deterministic_id",
                "confidence": 1.0,
                "max_similarity": 1.0,
                "results": [point.payload for point in results]
            }

    # ==========================================================
    # 2️⃣ Deterministic Keyword Lookup
    # ==========================================================
    for keyword in POLICY_KEYWORDS:
        if keyword in query_lower:

            results, _ = qdrant.scroll(
                collection_name="policies_collection",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value=keyword)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            if results:
                return {
                    "retrieval_type": "deterministic_keyword",
                    "confidence": 0.95,
                    "max_similarity": 0.95,
                    "results": [point.payload for point in results]
                }

    # ==========================================================
    # 3️⃣ Semantic Fallback
    # ==========================================================

    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

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
            "policy_id": hit.payload.get("policy_id"),
            "type": hit.payload.get("type"),
            "similarity_score": round(hit.score, 4),
            "payload": hit.payload
        })

    return {
        "retrieval_type": "semantic",
        "confidence": round(max_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "results": retrieved_results
    }