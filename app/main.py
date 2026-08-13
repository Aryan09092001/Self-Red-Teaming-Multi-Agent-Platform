import asyncio  # background worker loop and fire-and-forget tasks
import uuid  # uuid4 for session ids and report ids
import logging  # structured JSON logging, configured below
import traceback  # full stack trace when a job blows up
from contextlib import asynccontextmanager  # lifespan is an async context manager
from datetime import datetime  # timestamp on JSON-format reports
from fastapi import FastAPI, HTTPException, Request, Depends  # Depends wires auth and rate limiting into routes
from fastapi.responses import Response, FileResponse  # raw PDF bytes and the static frontend
from fastapi.middleware.cors import CORSMiddleware  # lets the browser frontend call this API
from pydantic import BaseModel  # request body validation
import redis.asyncio as aioredis  # async Redis, cache + sessions + job stream

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',  # JSON lines, so CloudWatch can parse fields
)
logger = logging.getLogger(__name__)  # configured before the app imports, so their loggers inherit this

from app.config import Config  # ponytail: imports below the logging setup on purpose, so module-level logs are formatted
from app.pool import init_pool, close_pool  # Postgres connection pool lifecycle
from app.auth import require_api_key  # X-API-Key dependency
from app.cache import cache_get, cache_set  # semantic cache
from app.guardrails import validate_input, validate_output  # Bedrock safety checks, both directions
from app.memory import session_add, session_get, ltm_search, ltm_search_related, ltm_store, ltm_diff, db_migrate  # short and long-term memory
from app.queue import push_job, get_result, set_result, ensure_group, consume_jobs, ack_job  # Redis Stream job queue
from app.agents import build_graph, ResearchState  # the LangGraph pipeline
from app.output import generate_pdf, generate_json_report, get_report_diff  # output formats
from app.eval import evaluate_report, run_batch_evaluation, fetch_recent_topics  # LLM-as-judge scoring

config = Config()  # built at import, so a bad secret fails the container immediately
redis_client: aioredis.Redis = None  # ponytail: module globals set in lifespan, fine for one process, would need rework for multi-worker state
graph = None  # compiled LangGraph, built once in lifespan


async def _rate_limit(request: Request) -> None:  # PURPOSE: per-IP throttle, used as a route dependency; raises 429 when the caller is over budget
    client_ip = request.client.host  # ponytail: direct socket IP, put the real client IP here if a proxy or ALB sits in front
    key = f"ratelimit:{client_ip}"  # one counter per IP per window
    count = await redis_client.incr(key)  # atomic, so concurrent requests count correctly
    if count == 1:
        await redis_client.expire(key, config.rate_limit_window)  # start the window on the first hit, fixed not sliding
    if count > config.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


async def _worker_loop():  # PURPOSE: the background consumer, runs forever pulling jobs off the stream and starting them
    await ensure_group(redis_client, config)  # create the consumer group once before reading
    while True:
        try:
            jobs = await consume_jobs(redis_client, config)  # blocks up to 5s waiting for work
            for job in jobs:
                asyncio.create_task(_process_job(job["data"], job["msg_id"]))  # not awaited, so the loop keeps consuming while jobs run
        except Exception:
            await asyncio.sleep(1)  # ponytail: silent catch-all, log the exception if worker failures need diagnosing


async def _process_job(data: dict, msg_id: str):  # PURPOSE: the full job pipeline for one request, from cache lookup through to stored result
    job_id = data["job_id"]
    topic = data["topic"]
    session_id = data["session_id"]
    output_format = data.get("output_format", "text")  # .get, older queued jobs may predate this field
    log = logging.getLogger(f"job.{job_id[:8]}")  # per-job logger, makes one request easy to trace in logs
    try:
        log.info(f"Starting job for topic: {topic}")

        # Fetch session history before any branch — agent always receives it
        session_history = await session_get(redis_client, session_id)

        cached = await cache_get(redis_client, config, topic)  # TIER 1: semantic cache, cheapest path
        if cached:
            log.info("Cache hit")
            report_text = cached
            await ltm_store(config, topic, report_text, str(uuid.uuid4()))  # re-stored so recency reflects actual use
        else:
            ltm_hit = await ltm_search(config, topic)  # TIER 2: long-term memory, a recent near-identical report
            if ltm_hit:
                log.info("LTM hit")
                report_text = ltm_hit["report"]
                await ltm_store(config, topic, report_text, str(uuid.uuid4()))
            else:
                log.info("Running multi-agent pipeline")  # TIER 3: the expensive path, full research run
                # Find a related (not identical) previous report for the writer to reference
                ltm_context = await ltm_search_related(config, topic) or ""
                if ltm_context:
                    log.info("Found related LTM context for writer agent")
                state = ResearchState(  # the graph's starting blackboard
                    topic=topic,
                    session_id=session_id,
                    session_history=session_history,  # agent is now context-aware
                    ltm_context=ltm_context,           # writer builds on prior research
                    search_results=[],  # filled by the search node
                    summaries=[],  # filled by the summarize node
                    report="",  # filled by the write node
                    verified=False,  # set by the verify node
                    error="",
                    iterations=0,  # counted up per write pass, bounds the retry loop
                )
                final_state = await graph.ainvoke(state)  # runs search, summarize, write, verify, possibly looping
                report_text = final_state["report"]
                ok, reason = await validate_output(config, report_text)  # guardrail the model's OUTPUT before anyone sees it
                if not ok:
                    await set_result(redis_client, config, job_id, {"status": "blocked", "error": reason})
                    await ack_job(redis_client, config, msg_id)  # ponytail: also acked in finally, XACK is idempotent so the repeat is harmless
                    return  # blocked reports are never cached or stored
                await cache_set(redis_client, config, topic, report_text)  # only freshly generated, guardrail-passed reports get cached
                await ltm_store(config, topic, report_text, str(uuid.uuid4()))

        await session_add(redis_client, config, session_id, "assistant", report_text[:config.session_content_truncate])  # record the reply as a conversation turn
        diff = await ltm_diff(config, topic)  # what changed versus the previous report on this topic
        result: dict = {"status": "done", "topic": topic, "report": report_text, "diff": diff}

        # Per-query evaluation runs automatically on every job
        asyncio.create_task(evaluate_report(config, job_id, topic, report_text))  # not awaited, scoring must not delay the user's result

        if output_format == "pdf":
            pdf_bytes = generate_pdf(topic, report_text)
            result["pdf_base64"] = __import__("base64").b64encode(pdf_bytes).decode()  # ponytail: inline import, move `import base64` to the top
        elif output_format == "json":
            result["structured"] = generate_json_report(topic, report_text, job_id, datetime.utcnow())

        await set_result(redis_client, config, job_id, result)  # published where /result can find it
        log.info("Job completed successfully")
    except Exception as e:
        log.error(f"Job failed: {traceback.format_exc()}")  # full trace to logs
        await set_result(redis_client, config, job_id, {"status": "error", "error": str(e)})  # client gets an error status instead of polling forever
    finally:
        await ack_job(redis_client, config, msg_id)  # always ack, so a poisoned job cannot be redelivered in a loop


@asynccontextmanager
async def lifespan(app: FastAPI):  # PURPOSE: startup and shutdown hooks; everything before yield runs once at boot, everything after at shutdown
    global redis_client, graph
    redis_client = await aioredis.from_url(config.redis_url, decode_responses=True)  # decode_responses so values come back as str, not bytes
    await init_pool(config)  # Postgres pool must exist before db_migrate
    await db_migrate(config)  # idempotent schema and index creation
    graph = build_graph(config)  # compiled once, reused by every job
    app.state.config = config  # where require_api_key reads the expected key from
    asyncio.create_task(_worker_loop())  # ponytail: worker runs in the same process as the API, split into its own service if load demands
    yield  # app serves requests here
    await redis_client.aclose()  # shutdown: close Redis...
    await close_pool()  # ...then the Postgres pool


app = FastAPI(title="Research Agent API", lifespan=lifespan)  # the ASGI app uvicorn serves
@app.middleware("http")
async def handle_head_requests(request, call_next):
    # Convert HEAD -> GET so endpoints that only implement GET still respond to HEAD.
    # After getting the response, remove the body for HEAD semantics.
    if request.method == "HEAD":
        # temporarily change the method to GET for routing
        request.scope["method"] = "GET"
        response = await call_next(request)
        # ensure no body is returned for HEAD and adjust Content-Length
        response.body = b""
        response.headers["content-length"] = "0"
        return response
    return await call_next(request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: wide open, restrict to the real frontend origin before production
    allow_methods=["GET", "POST", "HEAD"],  # only the verbs this API actually uses
    allow_headers=["*"],  # must permit X-API-Key
)


class ResearchRequest(BaseModel):  # PURPOSE: request body for POST /research, validated by pydantic before the handler runs
    topic: str  # required, the only mandatory field
    session_id: str = ""  # empty means "start a new conversation"
    output_format: str = "text"  # "text", "pdf" or "json"


@app.get("/")
async def frontend():  # PURPOSE: serve the single-page UI at the root path
    return FileResponse("/app/index.html")  # absolute container path, set by the Dockerfile


@app.get("/health")
async def health():  # PURPOSE: liveness probe for the load balancer; always 200, the body carries the real status
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False  # ponytail: checks Redis only, Postgres could be down and this still says ok
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "error",
    }


@app.post("/research", dependencies=[Depends(require_api_key), Depends(_rate_limit)])  # auth first, then throttle
async def start_research(req: ResearchRequest):  # PURPOSE: main entry point, queue a research job and return immediately with a job_id
    ok, reason = await validate_input(config, req.topic)  # guardrail the USER'S input before anything is queued
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    session_id = req.session_id or str(uuid.uuid4())  # new conversation when the client sent none
    await session_add(redis_client, config, session_id, "user", req.topic)  # record the question as a turn
    job_id = await push_job(redis_client, config, req.topic, session_id, req.output_format)  # onto the stream, a worker picks it up
    return {"job_id": job_id, "session_id": session_id}  # client polls /result/{job_id} with these


@app.get("/result/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job_result(job_id: str):  # PURPOSE: polling endpoint, returns the finished result or a pending marker
    result = await get_result(redis_client, config, job_id)
    if result is None:
        return {"status": "pending"}  # ponytail: also what an expired or unknown job_id returns, client would wait forever
    return result


@app.get("/session/{session_id}", dependencies=[Depends(require_api_key)])
async def get_session(session_id: str):  # PURPOSE: read back a conversation's recent turns, so the UI can rebuild the thread
    messages = await session_get(redis_client, session_id)
    return {"session_id": session_id, "messages": messages}  # empty list once the session TTL has lapsed


@app.get("/diff/{topic}", dependencies=[Depends(require_api_key)])
async def report_diff(topic: str):  # PURPOSE: show how reports on a topic changed over time
    diff = await get_report_diff(config, topic)  # semantic match, so near-identical topics still compare
    return {"topic": topic, "diff": diff or "No previous report found."}


@app.get("/result/{job_id}/pdf", dependencies=[Depends(require_api_key)])
async def download_pdf(job_id: str):  # PURPOSE: download any finished report as a PDF, regardless of the format originally requested
    result = await get_result(redis_client, config, job_id)
    if not result or result.get("status") != "done":
        raise HTTPException(status_code=404, detail="Report not ready")  # covers pending, blocked and errored jobs
    pdf_bytes = generate_pdf(result.get("topic", "Report"), result["report"])  # rendered on demand, not stored
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={job_id}.pdf"},  # forces a download rather than inline display
    )


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def stats():  # PURPOSE: operational snapshot of Redis usage and which backends are wired up
    info = await redis_client.info()  # server-wide Redis metrics
    keys = await redis_client.dbsize()  # total key count
    cache_keys = len([k async for k in redis_client.scan_iter("semantic:*")])  # ponytail: full scans, fine for a dashboard, avoid on a hot path
    session_keys = len([k async for k in redis_client.scan_iter("session:*")])
    return {
        "redis": {
            "total_keys": keys,
            "cache_entries": cache_keys,
            "active_sessions": session_keys,
            "memory_used_mb": round(info["used_memory"] / 1024 / 1024, 2),  # bytes to MB
            "connected_clients": info["connected_clients"],
            "uptime_hours": round(info["uptime_in_seconds"] / 3600, 1),
        },
        "tensorzero_url": config.tensorzero_url,  # ponytail: exposes internal URL and guardrail id behind auth, drop if the key is widely shared
        "guardrail_id": config.bedrock_guardrail_id,
    }


@app.get("/evaluate/{job_id}", dependencies=[Depends(require_api_key)])
async def evaluate_job(job_id: str):  # PURPOSE: re-score a finished report on demand and return the four judge scores
    result = await get_result(redis_client, config, job_id)
    if not result or result.get("status") != "done":
        raise HTTPException(status_code=404, detail="Job not done yet")
    scores = await evaluate_report(config, job_id, result["topic"], result["report"])  # awaited here, unlike the automatic run in _process_job
    return {"job_id": job_id, "topic": result["topic"], "scores": scores}


class BatchEvalRequest(BaseModel):  # PURPOSE: request body for POST /run-evaluation
    topics: list[str] = []  # empty means "use recent real topics from the database"


@app.post("/run-evaluation", dependencies=[Depends(require_api_key)])
async def trigger_batch_evaluation(req: BatchEvalRequest):  # PURPOSE: kick off a regression run over many topics in the background
    topics = req.topics if req.topics else await fetch_recent_topics()  # fall back to what users actually asked
    if not topics:
        raise HTTPException(status_code=400, detail="No topics found. Submit at least one research job first.")
    asyncio.create_task(run_batch_evaluation(config, graph, topics))  # ponytail: fire-and-forget, results only reach LangSmith, no way to poll progress
    return {"message": "Batch evaluation started in background", "topics": len(topics)}


# Purpose: the FastAPI entry point that wires every other module together and defines the HTTP
# surface. The API stays fast because it never researches inline: POST /research validates and
# guardrails the input, then queues a job and returns a job_id the client polls via /result.
# A background worker loop started in lifespan consumes that queue and runs _process_job, the
# real heart of the system, which tries three tiers in order (semantic cache, long-term memory,
# then the full multi-agent pipeline) before guardrailing the output, caching it, storing it and
# scoring it. lifespan owns all startup and shutdown: Redis, the Postgres pool, schema migration,
# the compiled graph and the worker. Remaining routes cover session replay, report diffs, PDF
# download, Redis stats and both on-demand and batch evaluation.
