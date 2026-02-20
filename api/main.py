from fastapi import FastAPI
from pydantic import BaseModel
import time
from dotenv import load_dotenv
import os
from openai import OpenAI

from router.intent_router import detect_intent
from retrievers.orders import retrieve_orders
from retrievers.products import retrieve_products
from retrievers.policies import retrieve_policies
from utils.logger import log_event

# Load environment once at startup
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Multi-Source RAG application is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/ask")
def ask(req: QueryRequest):
    start = time.time()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    intents = detect_intent(req.query)

    results = {}
    sources = []

    if "orders" in intents:
        results["orders"] = retrieve_orders(req.query)
        sources.append("orders_collection")

    if "products" in intents:
        results["products"] = retrieve_products(req.query)
        sources.append("products_collection")

    if "policies" in intents:
        results["policies"] = retrieve_policies(req.query)
        sources.append("policies_collection")

    context = str(results)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer using provided context only."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {req.query}"}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    latency = int((time.time() - start) * 1000)
    log_event(req.query, intents, sources, "success", latency)

    return {
        "answer": answer,
        "intents": intents,
        "sources": sources,
        "latency_ms": latency
    }
