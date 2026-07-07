from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from app.models import Document
from app.cleaner import clean_text


def load_txt(path: str) -> Document:
    raw_text = Path(path).read_text(
        encoding="utf-8"
    )

    cleaned_text = clean_text(raw_text)

    return Document(
        # Deterministic (not random uuid4): rebuilding the index from the
        # same source path yields the same document_id every time, which
        # downstream chunk_ids and labeled eval sets depend on.
        document_id=str(uuid5(NAMESPACE_URL, path)),
        source=path,
        text=cleaned_text,
    )