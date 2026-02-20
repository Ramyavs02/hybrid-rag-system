import json
import uuid
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_logs.json")

def log_event(query, intents, sources, status, latency):
    # ✅ Ensure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    log = {
        "log_id": f"LOG{uuid.uuid4().hex[:6].upper()}",
        "timestamp": datetime.utcnow().isoformat(),
        "user_query": query,
        "intent_detected": intents,
        "sources_used": sources,
        "response_status": status,
        "latency_ms": latency
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
