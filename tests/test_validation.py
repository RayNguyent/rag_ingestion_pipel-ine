import pytest
from pydantic import BaseModel

from app.registry.errors import MalformedJSONError, SchemaValidationError
from app.registry.validation import parse_and_validate


class DummyInput(BaseModel):
    query: str
    top_k: int = 5


def test_valid_json_string_parses_and_validates():
    result = parse_and_validate('{"query": "hello", "top_k": 3}', DummyInput)
    assert result == DummyInput(query="hello", top_k=3)


def test_dict_input_skips_json_decode():
    result = parse_and_validate({"query": "hello"}, DummyInput)
    assert result.query == "hello"


def test_trailing_comma_raises_malformed_json_error():
    with pytest.raises(MalformedJSONError) as exc_info:
        parse_and_validate('{"query": "hello",}', DummyInput)
    assert "not valid JSON" in exc_info.value.recovery_message


def test_unquoted_key_raises_malformed_json_error():
    with pytest.raises(MalformedJSONError):
        parse_and_validate('{query: "hello"}', DummyInput)


def test_missing_required_field_names_it_in_recovery_message():
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate('{"top_k": 3}', DummyInput)
    assert "query" in exc_info.value.recovery_message


def test_wrong_type_names_the_field_in_recovery_message():
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate('{"query": "hello", "top_k": "not-a-number"}', DummyInput)
    assert "top_k" in exc_info.value.recovery_message


def test_extra_unexpected_field_is_ignored_by_default_pydantic_config():
    # Pydantic's default is to ignore unknown fields rather than error —
    # documenting the current behavior so a future model_config change
    # (e.g. extra="forbid") is a deliberate choice, not a silent regression.
    result = parse_and_validate('{"query": "hello", "unexpected": true}', DummyInput)
    assert result.query == "hello"


def test_five_broken_payloads_each_produce_distinct_recovery_messages():
    broken_payloads = [
        '{"query": "hello",}',  # trailing comma
        '{query: "hello"}',  # unquoted key
        '{"top_k": 3}',  # missing required field
        '{"query": "hello", "top_k": "nope"}',  # wrong type
        "not json at all",  # not JSON
    ]
    messages = set()
    for payload in broken_payloads:
        with pytest.raises((MalformedJSONError, SchemaValidationError)) as exc_info:
            parse_and_validate(payload, DummyInput)
        messages.add(exc_info.value.recovery_message)

    assert len(messages) == len(broken_payloads)
