import time
import hashlib
import math
from collections import OrderedDict
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
CACHE_SIZE = 1500
TTL_SECONDS = 86400  # 24 hours
TOKEN_PER_REQUEST = 500
MODEL_COST_PER_1M = 0.50
SIMILARITY_THRESHOLD = 0.95

app = FastAPI()

# ================= CORS MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATA STRUCTURES =================
class CacheEntry:
    def __init__(self, query, answer, embedding):
        self.query = query
        self.answer = answer
        self.embedding = embedding
        self.timestamp = time.time()
        self.last_access = time.time()

cache = OrderedDict()

# ================= ANALYTICS =================
analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0,
}

# ================= UTILS =================
def normalize(text: str):
    return " ".join(text.lower().strip().split())

def hash_query(text: str):
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text: str):
    # Fake embedding (replace with real one in production)
    return [float(ord(c)) for c in text[:50]]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def remove_expired():
    now = time.time()
    keys_to_delete = []
    for key, entry in cache.items():
        if now - entry.timestamp > TTL_SECONDS:
            keys_to_delete.append(key)
    for k in keys_to_delete:
        del cache[k]

def enforce_lru():
    while len(cache) > CACHE_SIZE:
        cache.popitem(last=False)

def fake_llm_call(query):
    time.sleep(0.2)  # simulate API latency
    return f"Answer to: {query}"

# ================= REQUEST MODEL =================
class QueryRequest(BaseModel):
    query: str
    application: str

# ================= HEALTH CHECK =================
@app.get("/")
def health_check():
    return {"status": "ok"}

# ================= MAIN ENDPOINT =================
@app.post("/")
def handle_query(req: QueryRequest):
    start = time.time()

    # Handle empty or bad input
    if not req.query:
        return {
            "answer": "Empty query",
            "cached": False,
            "latency": 1,
            "cacheKey": "none"
        }

    analytics["totalRequests"] += 1

    normalized = normalize(req.query)
    key = hash_query(normalized)

    remove_expired()

    # 1. Exact match cache
    if key in cache:
        entry = cache[key]
        entry.last_access = time.time()
        cache.move_to_end(key)
        analytics["cacheHits"] += 1
        latency = max(1, int((time.time() - start) * 1000))

        return {
            "answer": str(entry.answer),
            "cached": True,
            "latency": latency,
            "cacheKey": str(key)
        }

    # 2. Semantic cache
    query_embedding = get_embedding(normalized)
    for k, entry in cache.items():
        sim = cosine_similarity(query_embedding, entry.embedding)
        if sim > SIMILARITY_THRESHOLD:
            entry.last_access = time.time()
            cache.move_to_end(k)
            analytics["cacheHits"] += 1
            latency = max(1, int((time.time() - start) * 1000))

            return {
                "answer": str(entry.answer),
                "cached": True,
                "latency": latency,
                "cacheKey": str(k)
            }

    # 3. Cache miss → call LLM
    analytics["cacheMisses"] += 1
    answer = fake_llm_call(normalized)

    # Store in cache
    cache[key] = CacheEntry(normalized, answer, query_embedding)
    cache.move_to_end(key)
    enforce_lru()

    latency = max(1, int((time.time() - start) * 1000))

    return {
        "answer": str(answer),
        "cached": False,
        "latency": latency,
        "cacheKey": str(key)
    }

# ================= ANALYTICS ENDPOINT =================
@app.get("/analytics")
def get_analytics():
    try:
        total = analytics.get("totalRequests", 0)
        hits = analytics.get("cacheHits", 0)
        misses = analytics.get("cacheMisses", 0)

        hit_rate = hits / total if total else 0

        baseline_tokens = total * TOKEN_PER_REQUEST
        actual_tokens = misses * TOKEN_PER_REQUEST

        baseline_cost = baseline_tokens * MODEL_COST_PER_1M / 1_000_000
        actual_cost = actual_tokens * MODEL_COST_PER_1M / 1_000_000
        savings = baseline_cost - actual_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost else 0

        return {
            "hitRate": round(hit_rate, 2),
            "totalRequests": int(total),
            "cacheHits": int(hits),
            "cacheMisses": int(misses),
            "cacheSize": int(len(cache)),
            "costSavings": round(savings, 2),
            "savingsPercent": int(savings_percent),
            "strategies": [
                "exact match",
                "semantic similarity",
                "LRU eviction",
                "TTL expiration"
            ]
        }
    except Exception:
        return {
            "hitRate": 0,
            "totalRequests": 0,
            "cacheHits": 0,
            "cacheMisses": 0,
            "cacheSize": 0,
            "costSavings": 0.0,
            "savingsPercent": 0,
            "strategies": [
                "exact match",
                "semantic similarity",
                "LRU eviction",
                "TTL expiration"
            ]
        }
