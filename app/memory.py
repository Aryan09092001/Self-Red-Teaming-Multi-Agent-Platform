import difflib  # unified_diff powers the report-vs-report comparison
import json  # session messages are stored in Redis as JSON strings
import asyncio  # to_thread, keeps CPU-bound embedding off the event loop
from datetime import datetime  # created_at timestamps on stored reports
import redis.asyncio as aioredis  # async Redis client, short-term session memory
from sentence_transformers import SentenceTransformer  # local embedding model, no API call
from app.config import Config  # TTLs, thresholds, index tuning
from app.pool import get_pool  # shared asyncpg pool, never opens its own connection

_model = SentenceTransformer("all-MiniLM-L6-v2")  # loaded once at import, 384-dim, must match vector(384) below


async def session_add(redis: aioredis.Redis, config: Config, session_id: str, role: str, content: str) -> None:  # PURPOSE: short-term memory, append one chat turn and keep only the recent window
    key = f"session:{session_id}"  # one Redis list per conversation
    await redis.rpush(key, json.dumps({"role": role, "content": content}))  # newest turn goes on the right
    await redis.ltrim(key, -config.session_max_messages, -1)  # keep the last N turns, drop older ones
    await redis.expire(key, config.session_ttl)  # refreshed on every turn, so idle sessions expire


async def session_get(redis: aioredis.Redis, session_id: str) -> list[dict]:  # PURPOSE: read back the recent turns of a conversation for prompt context
    messages = await redis.lrange(f"session:{session_id}", 0, -1)  # 0..-1 = whole list, oldest first
    return [json.loads(m) for m in messages]  # back to dicts, empty list if the session expired


async def db_migrate(config: Config) -> None:  # PURPOSE: create the reports table, pgvector extension and indexes; idempotent, run at startup
    pool = get_pool()
    async with pool.acquire() as conn:  # one connection, released back to the pool on exit
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # pgvector, needed for the vector column
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id         TEXT PRIMARY KEY,
                topic      TEXT NOT NULL,
                report     TEXT NOT NULL,
                embedding  vector(384),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)  # 384 dims must match the MiniLM model above
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS reports_embedding_idx
            ON reports USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {config.ivfflat_lists})
        """)  # approximate nearest-neighbour index, makes similarity search fast
        await conn.execute("CREATE INDEX IF NOT EXISTS reports_topic_idx ON reports (topic)")  # supports the exact-topic lookup in ltm_diff
        await conn.execute("CREATE INDEX IF NOT EXISTS reports_created_idx ON reports (created_at DESC)")  # supports the recency filters and ORDER BY


async def ltm_store(config: Config, topic: str, report: str, report_id: str) -> None:  # PURPOSE: long-term memory write, save a finished report with its topic embedding
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())  # encode blocks, so run it in a thread
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reports (id, topic, report, embedding, created_at)
            VALUES ($1, $2, $3, $4::vector, $5)
            ON CONFLICT (id) DO NOTHING
            """,  # ON CONFLICT makes a retried job harmless instead of an error
            report_id, topic, report, str(embedding), datetime.utcnow(),  # embedding cast from its text form to vector
        )


async def ltm_search(config: Config, topic: str) -> dict | None:  # PURPOSE: look for a recent report on essentially THIS topic, so the agent can reuse it instead of researching again
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, topic, report, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM reports
            WHERE created_at > NOW() - ($2 || ' days')::INTERVAL
              AND 1 - (embedding <=> $1::vector) > $3
            ORDER BY similarity DESC LIMIT 1
            """,  # <=> is pgvector cosine DISTANCE, so 1 - distance = similarity
            str(embedding), str(config.ltm_days), config.ltm_threshold,  # only the last N days, only above the high threshold
        )
        return dict(row) if row else None  # None = nothing close enough, do the research


async def ltm_search_related(config: Config, topic: str) -> str | None:  # PURPOSE: find a NEARBY earlier report to feed the writer as background, deliberately not an exact match
    """
    Finds a related (but not identical) previous report to use as reference context
    for the writer agent. Uses a lower threshold than ltm_search so it finds
    nearby topics rather than exact matches.
    """
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT report FROM reports
            WHERE 1 - (embedding <=> $1::vector) BETWEEN 0.5 AND $2
            ORDER BY created_at DESC LIMIT 1
            """,  # banded on purpose, 0.5 floor drops unrelated topics
            str(embedding), config.ltm_threshold - 0.01,  # ceiling just under ltm_search, so near-duplicates are excluded
        )
        return row["report"] if row else None  # None = no useful neighbour, writer starts clean


async def ltm_diff(config: Config, topic: str) -> str | None:  # PURPOSE: show what changed between the two latest reports on the same topic
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT report, created_at FROM reports WHERE topic = $1 ORDER BY created_at DESC LIMIT 2",  # exact topic match, newest two
            topic,
        )
        if len(rows) < 2:
            return None  # only one report so far, nothing to compare against
        old_lines = rows[1]["report"].splitlines(keepends=True)  # index 1 = the older of the two
        new_lines = rows[0]["report"].splitlines(keepends=True)  # index 0 = the newest
        diff_lines = list(difflib.unified_diff(  # standard +/- unified diff format
            old_lines, new_lines,
            fromfile=f"previous ({rows[1]['created_at'].date()})",  # dated headers make the diff readable
            tofile=f"latest ({rows[0]['created_at'].date()})",
            lineterm="",  # lines already carry their newline from keepends=True
        ))
        return "\n".join(diff_lines[:config.ltm_diff_limit * 10]) or "No significant changes detected."  # ponytail: cap is a crude line budget, replace if the truncation cuts mid-hunk


# Purpose: the agent's memory, in two layers. Short-term memory lives in Redis, one list per
# session_id holding the last few chat turns with a sliding TTL, so a conversation has context
# without growing forever. Long-term memory lives in Postgres with pgvector: every finished
# report is stored with an embedding of its topic, letting later questions be matched by
# meaning rather than exact wording. ltm_search reuses a recent near-identical report,
# ltm_search_related pulls a merely similar one as background for the writer, and ltm_diff
# compares the two newest reports on one topic to surface what actually changed over time.
