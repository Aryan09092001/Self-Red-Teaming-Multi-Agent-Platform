import json  # results are stored in Redis as JSON strings
import uuid  # uuid4 gives each job a collision-free id
import redis.asyncio as aioredis  # async Redis client, type hints only here
from app.config import Config  # supplies stream_key, consumer_group/name, result_ttl


async def push_job(redis: aioredis.Redis, config: Config, topic: str, session_id: str, output_format: str) -> str:  # PURPOSE: API side, drop a research request onto the stream and hand the caller a job_id to poll with
    job_id = str(uuid.uuid4())  # generated here, not by Redis, so the caller gets it immediately
    await redis.xadd(config.stream_key, {  # XADD appends to the stream, workers read from it
        "job_id": job_id,  # travels with the job so the worker knows where to write results
        "topic": topic,  # what to research
        "session_id": session_id,  # links the job to a conversation for session memory
        "output_format": output_format,  # requested report shape, e.g. markdown
    })
    return job_id  # returned to the client right away, work happens in the background


async def get_result(redis: aioredis.Redis, config: Config, job_id: str) -> dict | None:  # PURPOSE: poll for a finished job; returns the result dict, or None if still running or expired
    data = await redis.get(f"result:{job_id}")  # separate key space from the stream
    return json.loads(data) if data else None  # None covers both "not done yet" and "TTL expired"


async def set_result(redis: aioredis.Redis, config: Config, job_id: str, result: dict) -> None:  # PURPOSE: worker side, publish a finished job's result where the API can find it
    await redis.setex(f"result:{job_id}", config.result_ttl, json.dumps(result))  # SETEX so results self-clean after result_ttl


async def ensure_group(redis: aioredis.Redis, config: Config) -> None:  # PURPOSE: make sure the consumer group exists before a worker reads; safe to call on every worker start
    try:
        await redis.xgroup_create(config.stream_key, config.consumer_group, id="0", mkstream=True)  # id="0" reads backlog too, mkstream creates the stream if absent
    except Exception:
        pass  # ponytail: swallows BUSYGROUP (group already exists), which is the normal case; narrow to ResponseError if real errors need surfacing


async def consume_jobs(redis: aioredis.Redis, config: Config) -> list[dict]:  # PURPOSE: worker's main read, block up to 5s for the next unclaimed job and return it as a list
    messages = await redis.xreadgroup(  # group read, so each job goes to exactly one worker
        config.consumer_group,  # shared group name across all workers
        config.consumer_name,  # this worker's identity, hostname by default
        {config.stream_key: ">"},  # ">" = only messages never delivered to this group
        count=1,  # one job at a time, keeps work evenly spread
        block=5000,  # wait 5s for work instead of spinning in a hot loop
    )
    if not messages:
        return []  # timeout with no jobs, caller just loops again
    jobs = []
    for _, entries in messages:  # outer level is per stream, we only read one
        for msg_id, data in entries:  # inner level is the actual messages
            jobs.append({"msg_id": msg_id, "data": data})  # msg_id kept, ack_job needs it later
    return jobs


async def ack_job(redis: aioredis.Redis, config: Config, msg_id: str) -> None:  # PURPOSE: confirm a job finished so Redis stops tracking it as pending; without this it can be re-delivered
    await redis.xack(config.stream_key, config.consumer_group, msg_id)  # removes the message from this group's pending list


# Purpose: the asynchronous job queue that separates the API from the heavy research work,
# built on a Redis Stream instead of a plain list so delivery survives worker crashes. The
# API calls push_job and returns a job_id immediately; workers call ensure_group once, then
# loop on consume_jobs, and a consumer group guarantees each job is handed to exactly one
# worker. When the work is done the worker calls set_result and ack_job, and the client picks
# the answer up through get_result. Results carry a TTL so finished jobs clean themselves up.
