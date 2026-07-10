import time

import pytest
import requests

from app.registry.errors import (
    RateLimitedError,
    SchemaValidationError,
    TimeoutToolError,
    ToolCallError,
    TransientConnectionError,
)
from app.registry.resilience import call_with_resilience, classify_exception


def fake_sleep_recorder():
    """Returns (sleep_fn, calls) — records delays instead of actually waiting."""
    calls = []

    def _sleep(delay: float):
        calls.append(delay)

    return _sleep, calls


def test_succeeds_on_first_attempt_no_retry_needed():
    sleep, calls = fake_sleep_recorder()
    result = call_with_resilience(lambda: "ok", max_attempts=3, sleep=sleep)
    assert result == "ok"
    assert calls == []


def test_transient_error_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientConnectionError("connection reset")
        return "ok"

    sleep, calls = fake_sleep_recorder()
    result = call_with_resilience(flaky, max_attempts=5, sleep=sleep)

    assert result == "ok"
    assert attempts["n"] == 3
    assert len(calls) == 2  # slept before attempt 2 and attempt 3


def test_exhausting_retries_raises_the_retryable_error():
    def always_fails():
        raise TransientConnectionError("still down")

    sleep, calls = fake_sleep_recorder()
    with pytest.raises(TransientConnectionError):
        call_with_resilience(always_fails, max_attempts=3, sleep=sleep)

    assert len(calls) == 2  # slept between attempts 1->2 and 2->3, not after the last failure


def test_validation_error_never_retried():
    attempts = {"n": 0}

    def bad_args():
        attempts["n"] += 1
        raise SchemaValidationError("bad field 'query'")

    sleep, calls = fake_sleep_recorder()
    with pytest.raises(SchemaValidationError):
        call_with_resilience(bad_args, max_attempts=5, sleep=sleep)

    assert attempts["n"] == 1  # never retried
    assert calls == []


def test_timeout_raises_timeout_tool_error():
    def hangs():
        time.sleep(2)
        return "too late"

    sleep, calls = fake_sleep_recorder()
    with pytest.raises(TimeoutToolError):
        call_with_resilience(hangs, max_attempts=1, timeout_s=0.05, sleep=sleep)


def test_rate_limit_respects_retry_after_header():
    attempts = {"n": 0}

    def rate_limited_then_ok():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RateLimitedError(retry_after=2.5)
        return "ok"

    sleep, calls = fake_sleep_recorder()
    result = call_with_resilience(rate_limited_then_ok, max_attempts=3, sleep=sleep)

    assert result == "ok"
    assert calls == [2.5]  # used the server-provided delay, not computed backoff


# ---------------------------------------------------------------------------
# classify_exception: mapping raw `requests` exceptions onto our hierarchy
# ---------------------------------------------------------------------------
def test_classify_requests_timeout_as_transient():
    classified = classify_exception(requests.exceptions.Timeout("timed out"))
    assert isinstance(classified, TransientConnectionError)


def test_classify_requests_connection_error_as_transient():
    classified = classify_exception(requests.exceptions.ConnectionError("refused"))
    assert isinstance(classified, TransientConnectionError)


def test_classify_429_as_rate_limited_with_retry_after():
    response = requests.Response()
    response.status_code = 429
    response.headers["retry-after"] = "4"
    error = requests.exceptions.HTTPError(response=response)

    classified = classify_exception(error)

    assert isinstance(classified, RateLimitedError)
    assert classified.retry_after == 4.0


def test_classify_5xx_as_transient():
    response = requests.Response()
    response.status_code = 503
    error = requests.exceptions.HTTPError(response=response)

    classified = classify_exception(error)

    assert isinstance(classified, TransientConnectionError)


def test_classify_401_as_non_retryable():
    response = requests.Response()
    response.status_code = 401
    error = requests.exceptions.HTTPError(response=response)

    classified = classify_exception(error)

    assert isinstance(classified, ToolCallError)
    assert not isinstance(classified, (RateLimitedError, TransientConnectionError))


def test_call_with_resilience_never_retries_a_401_end_to_end():
    attempts = {"n": 0}

    def unauthorized():
        attempts["n"] += 1
        response = requests.Response()
        response.status_code = 401
        raise requests.exceptions.HTTPError(response=response)

    sleep, calls = fake_sleep_recorder()
    with pytest.raises(ToolCallError):
        call_with_resilience(unauthorized, max_attempts=5, sleep=sleep)

    assert attempts["n"] == 1
    assert calls == []


def test_retry_and_timeout_bounded_elapsed_time():
    """Regression guard: 3 attempts with small backoff should stay fast,
    not accidentally balloon (e.g. from a backoff bug multiplying delays)."""
    def always_fails():
        raise TransientConnectionError("down")

    start = time.perf_counter()
    with pytest.raises(TransientConnectionError):
        call_with_resilience(always_fails, max_attempts=3, base_delay=0.01)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0
