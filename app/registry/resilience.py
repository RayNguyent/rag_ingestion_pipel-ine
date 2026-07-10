import concurrent.futures
import random
import time
from typing import Callable, TypeVar

from app.registry.errors import (
    RateLimitedError,
    RetryableToolError,
    TimeoutToolError,
    ToolCallError,
    TransientConnectionError,
)

T = TypeVar("T")


def classify_exception(exc: Exception) -> Exception:
    """
    Maps a raw exception (e.g. from `requests`) onto our retryable
    hierarchy. Anything not recognized here is treated as fatal — it's
    safer to fail loudly on an unknown error than to silently retry
    something that will never succeed.

    Imports `requests` lazily so this module has no hard dependency on it
    for callers/tests that never touch network-calling tools.
    """
    try:
        import requests
    except ImportError:
        requests = None  # type: ignore

    if requests is not None:
        if isinstance(exc, requests.exceptions.Timeout):
            return TransientConnectionError(f"request timed out ({exc})")
        if isinstance(exc, requests.exceptions.ConnectionError):
            return TransientConnectionError(str(exc))
        if isinstance(exc, requests.exceptions.HTTPError):
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                retry_after = None
                if exc.response is not None:
                    header = exc.response.headers.get("retry-after")
                    if header is not None:
                        try:
                            retry_after = float(header)
                        except ValueError:
                            retry_after = None
                return RateLimitedError(retry_after=retry_after)
            if status is not None and 500 <= status < 600:
                return TransientConnectionError(f"upstream returned {status}")
            # 4xx other than 429 (auth, bad request to the upstream API, etc.)
            # is not retryable — the same call will fail the same way again.
            return ToolCallError(
                f"Underlying service call failed with a non-retryable HTTP "
                f"{status} error: {exc}"
            )

    # Already one of ours (e.g. raised directly inside a tool fn) — pass through.
    if isinstance(exc, ToolCallError):
        return exc

    # Unknown exception type: don't guess, treat as fatal.
    return ToolCallError(f"Tool call failed with an unexpected error: {exc}")


def call_with_resilience(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    timeout_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """
    Runs fn() with a hard timeout, retrying only exceptions classified as
    retryable, with exponential backoff + jitter. Non-retryable
    ToolCallErrors (bad args, permission denied, non-retryable HTTP
    errors) propagate on the first attempt — retrying those would just
    waste attempts on something only the caller can fix.

    `sleep` is injectable so tests can assert retry behavior without
    actually waiting.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return _run_with_timeout(fn, timeout_s, attempt)
        except RetryableToolError as e:
            last_error = e
            if attempt == max_attempts:
                raise
            delay = e.retry_after if isinstance(e, RateLimitedError) and e.retry_after else None
            if delay is None:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay * 0.3)
            sleep(delay)
        except ToolCallError:
            # Non-retryable by definition (validation, permission, non-retryable HTTP) — re-raise immediately.
            raise
        except Exception as e:
            # Anything else (e.g. a raw requests exception raised from inside
            # fn without going through classify_exception already) gets
            # classified here so the same retry/no-retry logic applies.
            classified = classify_exception(e)
            if isinstance(classified, RetryableToolError):
                last_error = classified
                if attempt == max_attempts:
                    raise classified from e
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay * 0.3)
                sleep(delay)
            else:
                raise classified from e

    # Unreachable in practice (loop always returns or raises), but keeps
    # type checkers happy and guards against a future logic change.
    raise last_error or ToolCallError("Tool call failed with no recorded error")


def _run_with_timeout(fn: Callable[[], T], timeout_s: float, attempt: int) -> T:
    # Deliberately not a `with` block: ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which blocks until the submitted thread
    # finishes — defeating the timeout entirely for a call that hangs
    # (the caller would still wait the full hang duration even though
    # TimeoutToolError already "fired" logically). shutdown(wait=False)
    # lets this function actually return at timeout_s; the orphaned
    # thread finishes in the background and is garbage collected after.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        raise TimeoutToolError(timeout_s, attempt) from None
    finally:
        executor.shutdown(wait=False)
