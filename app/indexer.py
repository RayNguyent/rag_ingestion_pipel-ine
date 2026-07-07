from uuid import NAMESPACE_URL, uuid5
from app.models import Chunk
import json

def create_chunk_records(
    document_id: str,
    source: str,
    chunks: list[str],
    tenant_id: str,
    acl_roles: list[str] | None = None,
    extra_metadata: dict | None = None,
):
    """
    Create a Chunk record with a unique chunk_id.

    Args:
        document_id (str): The ID of the document this chunk belongs to.
        source (str): Path/identifier of the source document.
        chunks (list[str]): The chunk texts.
        tenant_id (str): Owning tenant — enforced at every retrieval layer.
        acl_roles (list[str] | None): Roles allowed to see these chunks.
            Defaults to ["*"] (public within the tenant) if not given.
        extra_metadata (dict | None): Any additional filterable metadata
            (e.g. doc_type, department) to merge into each chunk's metadata.

    Returns:
        list[Chunk]: New Chunk records with unique chunk_ids and access metadata.
    """
    records = []
    for idx, chunk_text in enumerate(chunks):
        # Deterministic id (not random uuid4): rebuilding the index from the
        # same source produces the same chunk_ids, which a labeled eval set
        # depends on to stay valid across rebuilds.
        chunk_id = str(uuid5(NAMESPACE_URL, f"{document_id}:{idx}"))
        metadata = {
            "source": source,
            "chunk_number": idx + 1,
            "chunk_length": len(chunk_text),
            "tenant_id": tenant_id,
            "acl_roles": acl_roles or ["*"],
            **(extra_metadata or {}),
        }
        chunk_record = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=chunk_text,
            metadata=metadata
        )
        records.append(chunk_record)
        
    return records
    
def save_records(chunks, output_file: str):
    """
    Save a list of Chunk records to a JSON file.

    Args:
        chunks (list[Chunk]): A list of Chunk records to save.
        output_file (str): The path to the output JSON file.
    """
    data = [chunk.model_dump() for chunk in chunks]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)