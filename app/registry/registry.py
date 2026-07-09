from app.registry.models import ToolKind, ToolNotFoundError, ToolSpec


class ToolRegistry:
    """
    The single place tools get registered and looked up. Nothing in the
    codebase should call a tool's underlying function directly once it's
    registered here — always go through app/registry/execution.py's
    execute()/monitored_execute(), which is the only code path that runs
    the permission check.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"No tool registered under name '{name}'") from None

    def read_tools(self) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.kind == ToolKind.READ]

    def write_tools(self) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.kind == ToolKind.WRITE]

    def all_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def anthropic_tool_schemas(self, kinds: set[ToolKind] | None = None) -> list[dict]:
        """
        Build the `tools=[...]` list for an Anthropic API call, restricted
        to the given kinds. Pass kinds={ToolKind.READ} when handing tools
        to an LLM-driven agent loop so it can never even see write tools
        in its own tool list — the segregation is enforced before the
        model gets a chance to call anything.
        """
        wanted = kinds or {ToolKind.READ, ToolKind.WRITE}
        return [
            {
                "name": spec.name,
                "description": spec.description or spec.name,
                "input_schema": spec.input_model.model_json_schema(),
            }
            for spec in self._tools.values()
            if spec.kind in wanted
        ]


# Process-wide singleton. Individual tests can still construct their own
# ToolRegistry() instance for isolation.
registry = ToolRegistry()
