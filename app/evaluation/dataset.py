import json

from pydantic import BaseModel

from app.models import TenantContext


class EvalExample(BaseModel):
    query: str
    tenant_id: str
    user_id: str = "eval-user"
    roles: list[str] = ["*"]
    relevant_chunk_ids: list[str]

    def tenant_context(self) -> TenantContext:
        return TenantContext(tenant_id=self.tenant_id, user_id=self.user_id, roles=self.roles)


def load_eval_set(path: str) -> list[EvalExample]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EvalExample(**item) for item in data]
