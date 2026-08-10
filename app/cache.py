import json  # embeddings are stored in Redis as JSON arrays
import numpy as np  # vector math for the cosine similarity check
import redis.asyncio as aioredis  # async Redis client, type hints only here
from sentence_transformers import SentenceTransformer  # local embedding model, no API call
from app.config import Config  # supplies cache_ttl and cache_similarity_threshold

_model = SentenceTransformer("all-MiniLM-L6-v2")  # loaded once at import, 384-dim, small and fast
_CACHE_PREFIX = "semantic:"  # Redis key holding the cached answer text
_EMB_PREFIX = "emb:"  # parallel key holding that query's embedding


def _cosine_similarity(a: list, b: list) -> float:  # PURPOSE: score how alike two embeddings are, 1.0 = identical meaning, 0 = unrelated
    va, vb = np.array(a), np.array(b)  # lists to arrays so numpy ops apply
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))  # dot product over the product of magnitudes


def _embed(text: str) -> list:  # PURPOSE: turn text into a 384-number vector; one place to embed so get and set always agree
    return _model.encode(text).tolist()  # ndarray to list, needed for json.dumps


async def cache_get(redis: aioredis.Redis, config: Config, query: str) -> str | None:  # PURPOSE: look for an earlier answer to a question that MEANS the same as this one; returns the answer on a hit, None on a miss
    query_emb = _embed(query)  # embed the incoming question once
    async for key in redis.scan_iter(f"{_EMB_PREFIX}*"):  # ponytail: linear scan of all embeddings, move to a vector index if key count grows
        stored_emb = json.loads(await redis.get(key))  # decode the stored vector
        if _cosine_similarity(query_emb, stored_emb) >= config.cache_similarity_threshold:  # near-duplicate question, default 0.85
            cache_key = key.replace(_EMB_PREFIX, _CACHE_PREFIX)  # emb:123 -> semantic:123
            return await redis.get(cache_key)  # the cached answer for that earlier query
    return None  # nothing similar enough, caller runs the full agent


async def cache_set(redis: aioredis.Redis, config: Config, query: str, result: str) -> None:  # PURPOSE: save a finished answer plus its query embedding so future similar questions can hit the cache
    key_suffix = abs(hash(query))  # ponytail: PYTHONHASHSEED makes this vary per process, switch to sha256 if keys must be stable
    await redis.setex(f"{_CACHE_PREFIX}{key_suffix}", config.cache_ttl, result)  # answer, expires after cache_ttl
    await redis.setex(f"{_EMB_PREFIX}{key_suffix}", config.cache_ttl, json.dumps(_embed(query)))  # matching embedding, same TTL so both expire together


# Purpose: semantic cache in front of the research agent, so questions that mean the same
# thing reuse an existing answer instead of paying for another LLM run. Exact-match caching
# would miss paraphrases, so each query is embedded locally with all-MiniLM-L6-v2 and
# compared by cosine similarity: anything at or above cache_similarity_threshold counts as a
# hit. Every entry is stored as two Redis keys sharing one suffix, semantic:<id> for the
# answer and emb:<id> for its vector, both written with the same TTL so they expire in step.
