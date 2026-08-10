import asyncio  # to_thread, keeps the blocking boto3 call off the event loop
import boto3  # AWS SDK, used here for the bedrock-runtime client
from app.config import Config  # supplies region, guardrail id/version, retry settings
from app.retry import with_retry  # shared exponential-backoff helper


def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:  # PURPOSE: raw Bedrock call that scans one piece of text; blocking, never call directly from async code
    client = boto3.client("bedrock-runtime", region_name=config.aws_region)  # boto3 clients are not async, hence the thread
    return client.apply_guardrail(  # scans text against the configured Bedrock guardrail
        guardrailIdentifier=config.bedrock_guardrail_id,  # which guardrail policy to apply
        guardrailVersion=config.bedrock_guardrail_version,  # pinned version, so behaviour is reproducible
        source=source,  # "INPUT" or "OUTPUT", policies differ per direction
        content=[{"text": {"text": text}}],  # API takes a list of content blocks
    )


async def validate_input(config: Config, text: str) -> tuple[bool, str]:  # PURPOSE: screen the USER'S prompt before any LLM call; returns (allowed, reason)
    response = await with_retry(  # transient AWS errors should not block a legit request
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "INPUT"),  # lambda so each retry makes a fresh coroutine
        max_retries=config.llm_max_retries,  # same retry budget as LLM calls
        delay=config.llm_retry_delay,  # base backoff seconds
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":  # guardrail matched a blocked category
        return False, "Input blocked by safety guardrail."  # generic message, no policy details leaked to the caller
    return True, ""  # clean, empty reason


async def validate_output(config: Config, text: str) -> tuple[bool, str]:  # PURPOSE: screen the MODEL'S answer before it reaches the user; same (allowed, reason) contract
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "OUTPUT"),  # OUTPUT source, output-side policies
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":  # model produced disallowed content
        return False, "Output blocked by safety guardrail."  # drop the text, return the refusal instead
    return True, ""


# Purpose: safety layer between users and the model, backed by AWS Bedrock Guardrails.
# validate_input screens the user's prompt before any LLM call (prompt injection, harmful
# requests); validate_output screens the generated report before it reaches the user. Both
# return (allowed, reason) so callers just branch on a bool instead of parsing AWS responses.
# The boto3 SDK is synchronous, so the call is pushed to a worker thread via asyncio.to_thread
# and wrapped in with_retry, keeping the event loop free and surviving transient AWS errors.
