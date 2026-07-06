from uuid import uuid4
from app.models import Chunk
import json

def create_chunk_records(
    document_id: str, source: str, chunks: list[str]
):
    """
    Create a Chunk record with a unique chunk_id.

    Args:
        document_id (str): The ID of the document this chunk belongs to.
        text (str): The text content of the chunk.
        metadata (dict): Additional metadata for the chunk.

    Returns:
        Chunk: A new Chunk record with a unique chunk_id.
    """
    records = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = str(uuid4())
        metadata = {
            "source": source,
            "chunk_number": idx + 1,
            "chunk_length": len(chunk_text),
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