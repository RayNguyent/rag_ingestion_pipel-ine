from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    source: str
    text: str


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str
    text: str
    metadata: dict