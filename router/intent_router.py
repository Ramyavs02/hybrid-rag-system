import re

# ----------------------------
# REGEX PATTERNS (Deterministic Layer)
# ----------------------------

ORDER_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)
PRODUCT_PATTERN = re.compile(r"\bP\d+\b", re.IGNORECASE)

# ----------------------------
# KEYWORD RULES
# ----------------------------

POLICY_KEYWORDS = [
    "refund",
    "return",
    "warranty",
    "policy",
    "compensation",
    "shipping",
    "cancellation",
    "exchange",
    "privacy",
    "payment",
    "damaged",
    "defective",
    "broken",
    "late delivery",
    "delay",
    "lost"
]

PRODUCT_CONTEXT_KEYWORDS = [
    "price",
    "cost",
    "under",
    "above",
    "below",
    "available",
    "list",
    "show products"
]

INTENT_PRIORITY = ["orders", "products", "policies"]


# ----------------------------
# ENTERPRISE-GRADE ROUTER
# ----------------------------

def detect_intent(query: str) -> list[str]:
    query_lower = query.lower()
    intents = []

    # -------------------------
    # 1️⃣ Deterministic ID Detection (Highest Priority)
    # -------------------------
    if ORDER_PATTERN.search(query):
        intents.append("orders")

    if PRODUCT_PATTERN.search(query):
        intents.append("products")

    # -------------------------
    # 2️⃣ Policy Detection
    # -------------------------
    if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
        intents.append("policies")

    # -------------------------
    # 3️⃣ Contextual Product Detection
    # Trigger only when price/listing/filter context exists
    # -------------------------
    if any(word in query_lower for word in PRODUCT_CONTEXT_KEYWORDS):
        intents.append("products")

    # -------------------------
    # 4️⃣ Remove duplicates
    # -------------------------
    intents = list(set(intents))

    # -------------------------
    # 5️⃣ Stable Ordering (Enterprise Behavior)
    # -------------------------
    intents = sorted(intents, key=lambda x: INTENT_PRIORITY.index(x))

    return intents
