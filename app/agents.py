import asyncio  # ponytail: no longer referenced anywhere in this file, safe to delete
import httpx  # async HTTP client used to call the TensorZero gateway
import logging  # progress logging per agent step
from typing import TypedDict  # gives the LangGraph state a checked shape
from langgraph.graph import StateGraph, END, START  # graph builder plus the two sentinel nodes
from langsmith import traceable  # decorator that reports each step as a span in LangSmith
from app.config import Config  # gateway URL, retry budget, truncation limits
from app.retry import with_retry  # shared exponential-backoff helper

logger = logging.getLogger(__name__)  # module-scoped logger, named "app.agents"


class ResearchState(TypedDict):  # PURPOSE: the shared blackboard every node reads from and writes back into
    topic: str  # what the user asked about, set at graph entry
    session_id: str  # links this run to a conversation
    session_history: list[dict]  # prior conversation turns passed into agent
    ltm_context: str             # related previous report passed to writer
    search_results: list[str]  # raw findings from SearchAgent
    summaries: list[str]  # condensed bullets from SummarizeAgent
    report: str  # the drafted report from WriterAgent
    verified: bool  # CriticAgent's verdict, drives the retry decision
    error: str  # populated when a run fails
    iterations: int  # write passes so far, bounded by agent_max_iterations


async def _tz_call(config: Config, function_name: str, message: str) -> str:  # PURPOSE: the one entry point every agent uses to talk to the LLM, with retries applied
    return await with_retry(
        lambda: _tz_call_once(config, function_name, message),  # lambda so each retry builds a fresh coroutine
        max_retries=config.llm_max_retries,  # attempts before the error propagates
        delay=config.llm_retry_delay,  # base backoff seconds, doubles each failure
    )


async def _tz_call_once(config: Config, function_name: str, message: str) -> str:  # PURPOSE: one single unretried HTTP call to TensorZero; returns the model's text
    async with httpx.AsyncClient(timeout=120) as client:  # ponytail: new client per call, hoist to a shared one if latency matters
        response = await client.post(
            f"{config.tensorzero_url}/inference",  # gateway endpoint, handles model routing and observability
            json={
                "function_name": function_name,  # named TensorZero function, prompt and model live in its config
                "input": {"messages": [{"role": "user", "content": message}]},  # single-turn payload
            },
        )
        response.raise_for_status()  # turns 4xx/5xx into an exception so with_retry can see it
        return response.json()["content"][0]["text"]  # unwrap the first content block's text


class SearchAgent:  # PURPOSE: agent 1 of 4, the fact gatherer
    """Finds key facts. Receives session history so it understands what the user has asked before."""

    def __init__(self, config: Config):  # PURPOSE: hold the config, nothing else, agents are stateless between runs
        self.config = config

    @traceable(run_type="tool", name="agent:search")  # shows up as its own span in LangSmith
    async def run(self, topic: str, session_history: list[dict]) -> str:  # PURPOSE: STEP 1 of the pipeline, gather raw facts about the topic
        logger.info(f"SearchAgent: researching '{topic}'")

        history_ctx = ""  # stays empty for a brand-new conversation
        if session_history:
            recent = session_history[-4:]  # last 4 turns for context
            history_ctx = "\n\nPrevious conversation context (use this to understand what the user already knows and what angle they care about):\n"
            history_ctx += "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)  # USER:/ASSISTANT: transcript

        return await _tz_call(
            self.config,
            "research_summarize",  # TensorZero function name, shared with the summarize step
            f"You are a research specialist. Find and list 5 key facts, recent developments, "
            f"and important details about: {topic}. Be thorough and specific."
            f"{history_ctx}",  # appended only when there is prior conversation
        )


class SummarizeAgent:  # PURPOSE: agent 2 of 4, the condenser
    """Condenses raw search results into structured bullet points."""

    def __init__(self, config: Config):  # PURPOSE: hold the config for the LLM call below
        self.config = config

    @traceable(run_type="tool", name="agent:summarize")
    async def run(self, search_results: list[str]) -> str:  # PURPOSE: STEP 2, squeeze the raw findings down to structured bullets
        logger.info("SummarizeAgent: condensing search results")
        combined = "\n\n".join(search_results)  # blank line between results keeps them distinguishable
        return await _tz_call(
            self.config,
            "research_summarize",
            f"Summarize these research findings into clear, structured bullet points:\n\n{combined}",
        )


class WriterAgent:  # PURPOSE: agent 3 of 4, the report author
    """
    Produces the final structured report. If a related previous report exists in LTM,
    it uses it as reference so the new report builds on existing knowledge rather than
    starting from scratch.
    """

    def __init__(self, config: Config):  # PURPOSE: hold the config for the LLM call below
        self.config = config

    @traceable(run_type="tool", name="agent:writer")
    async def run(self, topic: str, summaries: list[str], ltm_context: str) -> str:  # PURPOSE: STEP 3, turn bullets plus past knowledge into the finished report
        logger.info("WriterAgent: drafting report")
        combined = "\n\n".join(summaries)

        ltm_section = ""  # omitted entirely when long-term memory found nothing related
        if ltm_context:
            ltm_section = (
                f"\n\nPREVIOUS RESEARCH ON A RELATED TOPIC (use this as reference — "
                f"build on it, correct outdated information, and highlight what has changed):\n"
                f"{ltm_context[:2000]}"  # hard cap, keeps the prompt from blowing up on long reports
            )

        return await _tz_call(
            self.config,
            "report_write",  # separate TensorZero function, likely a stronger model
            f"Write a comprehensive, well-structured research report on: '{topic}'\n\n"
            f"Current research findings:\n{combined}"
            f"{ltm_section}\n\n"
            f"Include: Executive Summary, Key Findings, Analysis, and Conclusion.",  # fixed section list keeps output consistent
        )


class CriticAgent:  # PURPOSE: agent 4 of 4, the quality gate
    """Verifies factual consistency and logical coherence of the report."""

    def __init__(self, config: Config):  # PURPOSE: hold the config for the LLM call below
        self.config = config

    @traceable(run_type="tool", name="agent:critic")
    async def run(self, report: str) -> bool:  # PURPOSE: STEP 4, quality gate; True = ship it, False = send the pipeline round again
        logger.info("CriticAgent: verifying report")
        check = await _tz_call(
            self.config,
            "research_summarize",
            f"Review this report for factual consistency and logical coherence. "
            f"Reply with YES if it passes or NO with a brief reason:\n\n"
            f"{report[:self.config.agent_report_truncate]}",  # truncated to bound token cost, default 3000 chars
        )
        return check.strip().upper().startswith("YES")  # ponytail: anything not starting with YES counts as a fail, deliberately strict


class OrchestratorAgent:  # PURPOSE: the conductor, owns all four agents and the retry decision
    """
    Coordinates all sub-agents. Passes session history to SearchAgent and LTM context
    to WriterAgent so the pipeline is genuinely context-aware. If the critic rejects
    the report it loops back for another pass (up to agent_max_iterations times).
    """

    def __init__(self, config: Config):  # PURPOSE: build one instance of each sub-agent, reused for every node in the graph
        self.config = config
        self.search_agent = SearchAgent(config)
        self.summarize_agent = SummarizeAgent(config)
        self.writer_agent = WriterAgent(config)
        self.critic_agent = CriticAgent(config)

    @traceable(run_type="chain", name="orchestrator:search")
    async def search_node(self, state: ResearchState) -> dict:  # PURPOSE: graph node wrapping SearchAgent; returns only the state keys it changed
        result = await self.search_agent.run(state["topic"], state.get("session_history", []))  # .get so a missing history is not an error
        return {"search_results": [result]}  # ponytail: replaces rather than appends, so a retry discards the earlier search

    @traceable(run_type="chain", name="orchestrator:summarize")
    async def summarize_node(self, state: ResearchState) -> dict:  # PURPOSE: graph node wrapping SummarizeAgent
        summary = await self.summarize_agent.run(state["search_results"])
        return {"summaries": [summary]}

    @traceable(run_type="chain", name="orchestrator:write")
    async def write_node(self, state: ResearchState) -> dict:  # PURPOSE: graph node wrapping WriterAgent; also counts this write pass
        report = await self.writer_agent.run(
            state["topic"],
            state["summaries"],
            state.get("ltm_context", ""),  # empty string when no related report was found
        )
        return {"report": report, "iterations": state.get("iterations", 0) + 1}  # the counter route() checks against the cap

    @traceable(run_type="chain", name="orchestrator:verify")
    async def verify_node(self, state: ResearchState) -> dict:  # PURPOSE: graph node wrapping CriticAgent; records the pass/fail verdict
        verified = await self.critic_agent.run(state["report"])
        return {"verified": verified}

    def route(self, state: ResearchState) -> str:  # PURPOSE: the loop decision, sends the run back to search or ends it
        """Orchestrator decision: retry search or finish."""
        if not state["verified"] and state.get("iterations", 0) < self.config.agent_max_iterations:  # failed AND budget left
            logger.info(f"Critic rejected report — retrying (iteration {state['iterations']})")
            return "search"  # loop back to the top of the pipeline
        return END  # passed, or out of iterations, so the last report ships as-is


def build_graph(config: Config):  # PURPOSE: wire the four nodes into a runnable LangGraph and return the compiled app
    orchestrator = OrchestratorAgent(config)  # one orchestrator owns every sub-agent
    workflow = StateGraph(ResearchState)  # state type declared up front, nodes return partial updates

    workflow.add_node("search", orchestrator.search_node)  # node names are what route() returns
    workflow.add_node("summarize", orchestrator.summarize_node)
    workflow.add_node("write", orchestrator.write_node)
    workflow.add_node("verify", orchestrator.verify_node)

    workflow.add_edge(START, "search")  # entry point of the graph
    workflow.add_edge("search", "summarize")  # fixed linear path...
    workflow.add_edge("summarize", "write")
    workflow.add_edge("write", "verify")  # ...up to the critic, where it branches
    workflow.add_conditional_edges(
        "verify",  # branch source
        orchestrator.route,  # function whose return value picks the next node
        {"search": "search", END: END},  # the only two outcomes route() can produce
    )

    return workflow.compile()  # compiled once at startup, then invoked per job


# Purpose: the multi-agent research pipeline itself, built as a LangGraph state machine.
# Four specialist agents run in sequence over one shared ResearchState: SearchAgent gathers
# facts (aware of session history), SummarizeAgent condenses them, WriterAgent drafts the
# report (reusing related past research from long-term memory), and CriticAgent judges it.
# The orchestrator's route() closes the loop: a rejected report goes back to search, up to
# agent_max_iterations times, so quality is bounded by a retry budget rather than an open
# loop. Every LLM call goes through _tz_call to the TensorZero gateway with retries, and
# each step is @traceable, so a whole run is inspectable end to end in LangSmith.
