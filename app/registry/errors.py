"""
Every exception here carries a `recovery_message` written for a model to
read, not a human debugging a stack trace: what was wrong, and what to do
about it.

Split into two families that resilience.py treats differently:

  ToolCallError (non-retryable) — the caller (usually the model) needs to
  change something before trying again. Retrying the exact same call
  would just fail the same way.

  RetryableToolError (transient) — the request itself was fine; the
  environment wasn't ready (timeout, momentary rate limit, connection
  blip). Safe to retry the same call unchanged, with backoff.
"""


class ToolCallError(Exception):
    """Base class for anything that should become a tool_result error
    block fed back to the model, rather than crashing the agent loop."""

    def __init__(self, recovery_message: str):
        self.recovery_message = recovery_message
        super().__init__(recovery_message)


class ToolNotFoundRecoveryError(ToolCallError):
    """The model called a tool name that isn't registered."""


class MalformedJSONError(ToolCallError):
    """Tool call arguments weren't valid JSON at all."""


class SchemaValidationError(ToolCallError):
    """Arguments were valid JSON but failed the tool's Pydantic input model."""


class OutputContractError(ToolCallError):
    """The tool ran but returned something that failed its own output
    model — a bug in the tool, not something the model can fix by
    retrying, but still worth surfacing as a clean message rather than a
    stack trace."""


class RetryableToolError(ToolCallError):
    """Transient failure — safe to retry the same call unchanged."""


class TimeoutToolError(RetryableToolError):
    def __init__(self, timeout_s: float, attempt: int):
        super().__init__(
            f"Tool call timed out after {timeout_s}s (attempt {attempt}). "
            f"This is a transient issue, not a problem with your arguments."
        )


class RateLimitedError(RetryableToolError):
    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            "Rate limited by the underlying service — retrying automatically, "
            "no action needed from you."
        )


class TransientConnectionError(RetryableToolError):
    def __init__(self, detail: str):
        super().__init__(
            f"Transient connection error calling the underlying service: {detail}. "
            f"Retrying automatically."
        )
