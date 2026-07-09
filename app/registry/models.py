from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Type

from pydantic import BaseModel


class ToolKind(str, Enum):
    """
    READ tools are side-effect free: they never mutate the vector store,
    the BM25 index, or any external system. An agent (or a read-only role)
    can be allowed to call these freely, including in a loop.

    WRITE tools mutate state (index writes, deletes, config changes). These
    always require an explicit write scope and are logged more heavily —
    see app/registry/metrics.py.
    """

    READ = "read"
    WRITE = "write"


class ToolNotFoundError(KeyError):
    """Raised when execute()/monitored_execute() is asked for an unregistered tool name."""


@dataclass(frozen=True)
class ToolSpec:
    """
    A tool's full contract: how to validate its input, how to validate its
    output, what scope a caller needs to invoke it, and the underlying
    callable. Nothing outside app/registry/execution.py should ever call
    `fn` directly — that's what keeps permission checks from being
    accidentally bypassed.
    """

    name: str
    kind: ToolKind
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    required_scope: str
    fn: Callable[[BaseModel], Any]
    description: str = ""
