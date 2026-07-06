from pathlib import Path
from uuid import uuid4
from app.models import Document
from app.cleaner import clean_text


def load_txt(path: str) -> Document:
    raw_text = Path(path).read_text(
        encoding="utf-8"
    )

    cleaned_text = clean_text(raw_text)

    return Document(
        document_id=str(uuid4()),
        source=path,
        text=cleaned_text,
    )