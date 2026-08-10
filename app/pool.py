import asyncpg  # async Postgres driver, provides the built-in connection pool
from app.config import Config  # typed settings, supplies database_url and pool bounds

_pool: asyncpg.Pool | None = None  # module-level singleton, None until init_pool runs


async def init_pool(config: Config) -> None:  # PURPOSE: open the shared Postgres pool; call once during app/worker startup
    global _pool  # rebinding the module singleton, not a local
    _pool = await asyncpg.create_pool(  # opens min_size connections up front
        config.database_url,  # full Postgres DSN from Secrets Manager
        min_size=config.db_pool_min,  # connections kept warm, avoids cold-start handshakes
        max_size=config.db_pool_max,  # ceiling, keep under Postgres max_connections
    )


async def close_pool() -> None:  # PURPOSE: shut the pool down and free every connection; call on shutdown
    global _pool
    if _pool:  # idempotent, safe to call even if init_pool never ran
        await _pool.close()  # waits for in-flight queries, then closes every connection
        _pool = None  # reset so a later init_pool starts fresh


def get_pool() -> asyncpg.Pool:  # PURPOSE: hand out the live pool to query code; the accessor used everywhere instead of touching _pool
    if _pool is None:  # fail loud on a startup-order bug rather than on a None attribute
        raise RuntimeError("Database pool not initialized")
    return _pool  # caller does `async with get_pool().acquire() as conn:`


# Purpose: owns the single shared asyncpg connection pool for the whole process. Opening a
# Postgres connection per query is slow and would exhaust the server's connection limit, so
# one pool is created at startup (init_pool), reused by every request and worker via
# get_pool, and closed on shutdown (close_pool). The pool is a module-level singleton, which
# means init_pool must run before any query; get_pool raises RuntimeError instead of
# returning None so that ordering mistakes surface immediately with a clear message.
