import json

from pydantic import BaseModel, ValidationError

from app.registry.errors import MalformedJSONError, SchemaValidationError


def parse_and_validate(raw_args: str | dict, input_model: type[BaseModel]) -> BaseModel:
    """
    Two independent failure modes, handled separately because they need
    different recovery messages:

      1. raw_args isn't valid JSON at all (only relevant when raw_args is
         a str, e.g. straight off the wire from a model's tool_use block).
      2. raw_args parses fine but doesn't match the tool's input schema
         (wrong types, missing required fields, out-of-range values).
    """
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as e:
            raise MalformedJSONError(
                f"Your tool call arguments were not valid JSON: {e.msg} "
                f"at line {e.lineno}, column {e.colno}. Re-emit the call "
                f"with strictly valid JSON — check for trailing commas, "
                f"unquoted keys, or unescaped quotes."
            ) from e
    else:
        parsed = raw_args

    try:
        return input_model.model_validate(parsed)
    except ValidationError as e:
        field_errors = "; ".join(_format_field_error(err) for err in e.errors())
        raise SchemaValidationError(
            f"Your arguments failed validation against this tool's schema: "
            f"{field_errors}. Correct these fields and re-emit the call."
        ) from e


def _format_field_error(err: dict) -> str:
    field_path = ".".join(str(p) for p in err["loc"]) or "(top level)"
    return f"'{field_path}' — {err['msg']}"
