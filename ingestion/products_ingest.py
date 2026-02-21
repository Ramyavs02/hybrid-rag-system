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
COLLECTION_NAME = "products_collection"

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
print("Fetching products from S3...")

response = s3.get_object(
    Bucket=BUCKET,
    Key="products/products.json"
)

products = json.loads(response["Body"].read().decode("utf-8"))

print(f"Total products found: {len(products)}")

# ---------- PRODUCT TO TEXT ----------
def product_to_text(product):
    warranty_text = (
        f"It comes with a warranty of {product['warranty']}."
        if product.get("warranty")
        else "Warranty information is not available."
    )

    return (
        f"Product {product['name']} (ID {product['product_id']}) "
        f"belongs to the {product['category']} category. "
        f"It is priced at {product['price']} INR. "
        f"{product['description']} "
        f"{warranty_text}"
    )

# ---------- CREATE COLLECTION ----------
if not qdrant.collection_exists(COLLECTION_NAME):
    print("Creating products collection...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# ---------- GENERATE EMBEDDINGS ----------
points = []

print("Generating embeddings...")

for product in products:
    text = product_to_text(product)

    embedding = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    ).data[0].embedding

    points.append({
        "id": str(uuid.uuid4()),
        "vector": embedding,
        "payload": product
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

print(f"✅ Products ingestion completed successfully. Total: {len(points)}")