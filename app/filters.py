from typing import Any


def matches_filters(metadata: dict, filters: dict[str, Any] | None) -> bool:
    """
    Generic metadata filter matcher shared by the vector store and BM25 index,
    so tenant/ACL/metadata filtering behaves identically across both branches
    of the hybrid retriever.

    Filter semantics:
    - filters["tenant_id"] = "acme"          -> exact match required
    - filters["doc_type"] = ["faq", "policy"] -> value must be in the list
    - filters["acl_roles"] = ["viewer"]       -> caller's roles must overlap
        the chunk's acl_roles, OR the chunk is public (acl_roles contains "*")
    """
    if not filters:
        return True

    for key, condition in filters.items():
        if key == "acl_roles":
            chunk_roles = metadata.get("acl_roles", [])
            if "*" in chunk_roles:
                continue
            caller_roles = condition if isinstance(condition, (list, tuple, set)) else [condition]
            if not set(caller_roles).intersection(chunk_roles):
                return False
        elif isinstance(condition, (list, tuple, set)):
            if metadata.get(key) not in condition:
                return False
        else:
            if metadata.get(key) != condition:
                return False

    return True
