import json
import os
import uuid
import boto3
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

# ---------- ENV ----------
BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")
COLLECTION_NAME = "orders_collection"

# ---------- CLIENTS ----------
s3 = boto3.client("s3", region_name=REGION)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ QDRANT CLOUD CONNECTION
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# ---------- READ FROM S3 ----------
response = s3.get_object(
    Bucket=BUCKET,
    Key="orders/orders.json"
)

orders = json.loads(response["Body"].read().decode("utf-8"))

def order_to_text(order):
    return (
        f"Order {order['order_id']} placed on {order['order_date']}. "
        f"Order status: {order['order_status']}. "
        f"Payment status: {order['payment_status']}."
    )

# ---------- CREATE COLLECTION ----------
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# ---------- EMBEDDINGS ----------
points = []

for order in orders:
    text = order_to_text(order)

    embedding = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    ).data[0].embedding

    points.append({
        "id": str(uuid.uuid4()),
        "vector": embedding,
        "payload": order
    })

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ Orders ingestion completed. Total: {len(points)}")
