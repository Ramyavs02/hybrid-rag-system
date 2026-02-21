# core/retrieval_manager.py

import time
from typing import Dict, Any, List

from retrievers.orders import retrieve_orders
from retrievers.products import retrieve_products
from retrievers.policies import retrieve_policies


# ---------------------------------------------------
# Safe Wrapper (Prevents Backend Crashes)
# ---------------------------------------------------
def safe_call(func, query: str, intent: str) -> Dict[str, Any]:
    try:
        result = func(query)

        # Normalize structure safely
        return {
            "intent": intent,
            "retrieval_type": result.get("retrieval_type", "unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "results": result.get("results", []),
            "error": None
        }

    except Exception as e:
        return {
            "intent": intent,
            "retrieval_type": "error",
            "confidence": 0.0,
            "results": [],
            "error": str(e)
        }


# ---------------------------------------------------
# Main Retrieval Manager
# ---------------------------------------------------
def run_retrieval(query: str) -> Dict[str, Any]:

    start_time = time.time()

    # Call all retrievers safely
    orders_data = safe_call(retrieve_orders, query, "orders")
    products_data = safe_call(retrieve_products, query, "products")
    policies_data = safe_call(retrieve_policies, query, "policies")

    all_results = [orders_data, products_data, policies_data]

    aggregated_context: List[Dict[str, Any]] = []
    sources: List[str] = []
    confidences: List[float] = []
    retrieval_types: List[str] = []

    # Aggregate results
    for item in all_results:

        if item["results"]:
            aggregated_context.extend(item["results"])
            sources.append(item["intent"])
            confidences.append(item["confidence"])
            retrieval_types.append(item["retrieval_type"])

    overall_confidence = max(confidences) if confidences else 0.0

    # Faithfulness proxy (simple heuristic for now)
    faithfulness = round(overall_confidence, 4)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "aggregated_results": aggregated_context,
        "sources": list(set(sources)),
        "confidence": round(overall_confidence, 4),
        "faithfulness": faithfulness,
        "retrieval_types": list(set(retrieval_types)),
        "latency_ms": latency_ms,
        "debug": {
            "orders": orders_data,
            "products": products_data,
            "policies": policies_data
        }
    }
