from pydantic import BaseModel


def build_tool_result_block(tool_use_id: str, *, error: Exception | None, result: BaseModel | None = None) -> dict:
    """
    Shapes a tool outcome as an Anthropic API `tool_result` content block.
    Errors and successes use the same block shape (keyed to the same
    tool_use_id) so the agent loop's message-building code doesn't need
    an error-specific branch — only `is_error` differs.

    Accepts any Exception for `error`, not just ToolCallError — falls
    back to str(error) for exception types (e.g. PermissionDeniedError)
    that don't define `.recovery_message`.
    """
    if error is not None:
        message = getattr(error, "recovery_message", None) or str(error)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": message,
            "is_error": True,
        }

    if result is None:
        raise ValueError("build_tool_result_block requires either `error` or `result`")

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result.model_dump_json(),
    }
