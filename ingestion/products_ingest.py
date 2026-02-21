import os
import uuid
import boto3
import fitz
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- ENV ----------
BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")
COLLECTION_NAME = "policies_collection"

# ---------- CLIENTS ----------
s3 = boto3.client("s3", region_name=REGION)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ QDRANT CLOUD CONNECTION
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# ---------- CREATE COLLECTION ----------
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# ---------- LIST PDFs ----------
response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix="policies/"
)

points = []

for obj in response.get("Contents", []):
    key = obj["Key"]

    if not key.endswith(".pdf"):
        continue

    print(f"Processing {key}")

    pdf_obj = s3.get_object(Bucket=BUCKET, Key=key)
    pdf_bytes = pdf_obj["Body"].read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    chunks = text_splitter.split_text(full_text)

    for chunk in chunks:
        embedding = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        ).data[0].embedding

        points.append({
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "source": "policies",
                "document": key,
                "text": chunk
            }
        })

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ Policies ingestion completed. Total chunks: {len(points)}")
