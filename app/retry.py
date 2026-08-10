import asyncio  # sleep between attempts without blocking the event loop
import logging  # warn on each failed attempt so retries are visible in logs

logger = logging.getLogger(__name__)  # module-scoped logger, named "app.retry"


async def with_retry(coro_fn, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):  # coro_fn is a zero-arg callable, not an awaited coroutine
    """Call coro_fn() with exponential backoff. Raises the last exception if all retries fail."""
    last_exc = None  # holds the most recent failure, re-raised if every attempt fails
    wait = delay  # current sleep, multiplied by backoff after each failure
    for attempt in range(1, max_retries + 1):  # 1-indexed so log messages read naturally
        try:
            return await coro_fn()  # fresh coroutine per attempt, an awaited one cannot be retried
        except Exception as exc:  # broad on purpose: any transient error is worth retrying
            last_exc = exc  # remember it in case this was the final attempt
            if attempt < max_retries:  # skip the sleep after the last attempt
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}. Retrying in {wait:.1f}s")  # visibility into flaky dependencies
                await asyncio.sleep(wait)  # 1s, 2s, 4s... with the defaults
                wait *= backoff  # exponential growth, eases pressure on a struggling service
    raise last_exc  # all attempts exhausted, surface the real error to the caller


# Purpose: one generic retry helper for any async call that can fail transiently, mainly
# LLM gateway and network calls. Pass a zero-argument callable (a lambda or partial) so a
# brand-new coroutine is created on every attempt; passing an already-awaited coroutine
# would fail on the second try. Failures are retried with exponential backoff (delay,
# delay*backoff, delay*backoff^2, ...), each retry logged at WARNING level. If every
# attempt fails the original exception is re-raised, so callers see the true cause.
