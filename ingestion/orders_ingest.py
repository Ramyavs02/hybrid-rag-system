import json
import os
import uuid
import boto3
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ---------- LOAD ENV ----------
load_dotenv()

BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")
COLLECTION_NAME = "orders_collection"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------- CLIENTS ----------
s3 = boto3.client("s3", region_name=REGION)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# ---------- READ FROM S3 ----------
print("Fetching orders from S3...")

response = s3.get_object(
    Bucket=BUCKET,
    Key="orders/orders.json"
)

orders = json.loads(response["Body"].read().decode("utf-8"))

print(f"Total orders found: {len(orders)}")

# ---------- ORDER TO TEXT ----------
def order_to_text(order):
    return (
        f"Order {order['order_id']} placed on {order['order_date']}. "
        f"Order status: {order['order_status']}. "
        f"Payment status: {order['payment_status']}."
    )

# ---------- CREATE COLLECTION IF NOT EXISTS ----------
if not qdrant.collection_exists(COLLECTION_NAME):
    print("Creating collection...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# ---------- GENERATE EMBEDDINGS ----------
points = []

print("Generating embeddings...")

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

print(f"Embeddings created: {len(points)}")

# ---------- BATCH UPSERT ----------
BATCH_SIZE = 50

print("Uploading to Qdrant Cloud...")

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"Uploaded batch {i // BATCH_SIZE + 1}")

print(f"✅ Orders ingestion completed successfully. Total: {len(points)}")