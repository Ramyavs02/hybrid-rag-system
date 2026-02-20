import os
#from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

#load_dotenv()

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def retrieve_products(query: str, limit: int = 3):
    embedding = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding

    results = qdrant.query_points(
        collection_name="products_collection",
        query=embedding,
        limit=limit
    )

    return [p.payload for p in results.points]
