import asyncio  # gather, runs the four judges concurrently
import re  # pulls the numeric score out of the judge's reply
import httpx  # async HTTP client for the TensorZero gateway
import logging  # warn when LangSmith logging fails
from langsmith import Client, traceable  # Client writes eval examples, traceable makes each judge a span
from app.config import Config  # gateway URL, retry budget, truncation limits
from app.retry import with_retry  # shared exponential-backoff helper

logger = logging.getLogger(__name__)  # module-scoped logger, named "app.eval"


_ls_client: Client | None = None  # lazily created singleton, avoids building a client at import time


def _ls() -> Client:  # PURPOSE: hand back the one shared LangSmith client, creating it on first use
    global _ls_client
    if _ls_client is None:
        _ls_client = Client()  # reads LANGCHAIN_API_KEY from env, set by Config.__init__
    return _ls_client


def _parse_score(text: str) -> float:  # PURPOSE: pull "SCORE: 8/10" out of the judge's free text and normalise it to 0.0-1.0
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)  # tolerant of spacing and case, accepts decimals
    return round(float(m.group(1)) / 10.0, 2) if m else 0.5  # ponytail: unparseable reply silently scores 0.5, log it if bad judges need catching


async def _judge(config: Config, prompt: str) -> str:  # PURPOSE: the one entry point every judge uses to call the LLM, with retries applied
    return await with_retry(
        lambda: _judge_once(config, prompt),  # lambda so each retry builds a fresh coroutine
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )


async def _judge_once(config: Config, prompt: str) -> str:  # PURPOSE: one single unretried HTTP call to TensorZero; returns the judge's text
    async with httpx.AsyncClient(timeout=60) as client:  # shorter timeout than the agents, judging is a smaller job
        r = await client.post(
            f"{config.tensorzero_url}/inference",
            json={
                "function_name": "research_summarize",  # reuses the general-purpose function, no dedicated judge model
                "input": {"messages": [{"role": "user", "content": prompt}]},
            },
        )
        r.raise_for_status()  # turns 4xx/5xx into an exception so with_retry sees it
        return r.json()["content"][0]["text"]


@traceable(run_type="chain", name="eval:relevance")  # own span in LangSmith
async def eval_relevance(config: Config, topic: str, report: str) -> dict:  # PURPOSE: judge 1 of 4, does the report actually answer the topic asked
    verdict = await _judge(
        config,
        f"Rate how relevant this research report is to the topic '{topic}'.\n"
        f"Reply with exactly: SCORE: X/10 on the first line, then one sentence reason.\n\n"  # strict format so _parse_score can read it
        f"Report:\n{report[:config.eval_report_truncate]}",  # truncated to bound token cost
    )
    return {"key": "relevance", "score": _parse_score(verdict), "comment": verdict[:config.eval_comment_truncate]}  # HIGHER is better


@traceable(run_type="chain", name="eval:completeness")
async def eval_completeness(config: Config, report: str) -> dict:  # PURPOSE: judge 2 of 4, are all four required sections present
    verdict = await _judge(
        config,
        f"Does this research report contain all four required sections: "
        f"Executive Summary, Key Findings, Analysis, and Conclusion?\n"  # same section list WriterAgent was told to produce
        f"Reply with exactly: SCORE: X/10 on the first line, then one sentence reason.\n\n"
        f"Report:\n{report[:config.eval_report_truncate]}",
    )
    return {"key": "completeness", "score": _parse_score(verdict), "comment": verdict[:config.eval_comment_truncate]}  # HIGHER is better


@traceable(run_type="chain", name="eval:hallucination_risk")
async def eval_hallucination(config: Config, topic: str, report: str) -> dict:  # PURPOSE: judge 3 of 4, hunt for invented facts, fake stats and impossible dates
    verdict = await _judge(
        config,
        f"Check this report on '{topic}' for hallucinations — fabricated statistics, "
        f"impossible dates, or claims that contradict well-known facts.\n"
        f"Score: 1/10 = zero hallucinations detected, 10/10 = many hallucinations.\n"  # scale is INVERTED here, low is good
        f"Reply with exactly: SCORE: X/10 on the first line, then list any suspicious claims.\n\n"
        f"Report:\n{report[:config.eval_report_truncate]}",
    )
    return {"key": "hallucination_risk", "score": _parse_score(verdict), "comment": verdict[:config.eval_comment_truncate]}  # LOWER is better, unlike the other three


@traceable(run_type="chain", name="eval:overall_quality")
async def eval_quality(config: Config, topic: str, report: str) -> dict:  # PURPOSE: judge 4 of 4, holistic verdict on depth, accuracy, clarity and usefulness
    verdict = await _judge(
        config,
        f"Rate the overall quality of this research report on '{topic}'.\n"
        f"Consider: depth of analysis, factual accuracy, writing clarity, logical structure, "
        f"and practical usefulness to a business analyst.\n"  # named audience keeps the rating consistent between runs
        f"Reply with exactly: SCORE: X/10 on the first line, then two sentences explaining the rating.\n\n"
        f"Report:\n{report[:config.eval_report_truncate]}",
    )
    return {"key": "overall_quality", "score": _parse_score(verdict), "comment": verdict[:config.eval_comment_truncate]}  # HIGHER is better


@traceable(run_type="chain", name="evaluate-report")  # parent span, the four judges nest under it
async def evaluate_report(config: Config, job_id: str, topic: str, report: str) -> dict:  # PURPOSE: the main entry point, score one report on all four axes and log it to LangSmith
    """Runs all 4 LLM judges in parallel. Called on EVERY research job automatically."""
    results = await asyncio.gather(  # concurrent, so four judges cost about one judge's latency
        eval_relevance(config, topic, report),
        eval_completeness(config, report),
        eval_hallucination(config, topic, report),
        eval_quality(config, topic, report),
    )
    scores = {r["key"]: r["score"] for r in results}  # flatten to {"relevance": 0.8, ...}
    try:  # everything below is observability, it must never break a research job
        client = _ls()
        try:
            dataset = client.read_dataset(dataset_name=config.langsmith_dataset)  # reuse the dataset if it exists
        except Exception:
            dataset = client.create_dataset(  # first run, create it
                config.langsmith_dataset,
                description="Research agent LLM-as-judge evaluation results",
            )
        client.create_example(  # one row per evaluated job, so scores can be tracked over time
            inputs={"topic": topic},
            outputs={"report_preview": report[:400]},  # preview only, full reports live in Postgres
            dataset_id=dataset.id,
            metadata={"job_id": job_id, **scores},  # the four scores ride along as metadata
        )
    except Exception as e:
        logger.warning(f"LangSmith logging failed for job {job_id}: {e}")  # logged, not raised, scores are still returned
    return scores


async def fetch_recent_topics(limit: int = 10) -> list[str]:  # PURPOSE: build a batch-eval topic list out of what users actually asked
    """Pull distinct topics from the reports table — real user queries, nothing hardcoded."""
    from app.pool import get_pool  # imported here, not at module top, to avoid a circular import
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT topic FROM reports GROUP BY topic ORDER BY MAX(created_at) DESC LIMIT $1",  # distinct topics, most recently used first
            limit,
        )
        return [row["topic"] for row in rows]


async def run_batch_evaluation(config: Config, graph, topics: list[str]) -> list[dict]:  # PURPOSE: regression harness, re-run the whole pipeline over many topics and score each result
    from app.agents import ResearchState  # local imports again, keeps eval out of the agents import cycle
    from app.memory import ltm_search_related
    results = []
    for topic in topics:  # ponytail: sequential on purpose, running full pipelines in parallel would hammer the LLM gateway
        ltm_context = await ltm_search_related(config, topic) or ""  # same memory context a real run would get
        state = ResearchState(
            topic=topic, session_id="batch-eval",  # fixed session id marks these as non-user runs
            session_history=[],  # no conversation history in batch mode
            ltm_context=ltm_context,
            search_results=[], summaries=[], report="",  # empty starting state, the graph fills these
            verified=False, error="", iterations=0,
        )
        final = await graph.ainvoke(state)  # runs the full search-summarize-write-verify loop
        scores = await evaluate_report(config, f"batch-{topic[:20]}", topic, final["report"])  # synthetic job_id tags the batch in LangSmith
        results.append({"topic": topic, "scores": scores})
    return results


# Purpose: the automated quality layer, an LLM-as-judge system that scores every report the
# agent produces. Four independent judges run concurrently on each job: relevance, completeness,
# hallucination risk and overall quality. Each is asked for a strict "SCORE: X/10" line that
# _parse_score converts to 0.0-1.0. Note the scales differ, hallucination_risk is inverted so
# LOW is good, while the other three want HIGH. Results are pushed to a LangSmith dataset for
# tracking over time, wrapped in try/except so an observability outage never fails a user's job.
# run_batch_evaluation reuses the same judges as a regression harness over real past topics.
