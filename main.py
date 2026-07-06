from app.loader import load_txt
from app.chunker import split_text
from app.indexer import create_chunk_records, save_records

doc = load_txt("data/sample.txt")
chunks = split_text(doc.text, chunk_size=50, chunk_overlap=10)
records = create_chunk_records(document_id=doc.document_id, source=doc.source, chunks=chunks)
save_records(records, "output/chunks.json")
print("Chunking and saving completed. Check output/chunks.json for the results.")