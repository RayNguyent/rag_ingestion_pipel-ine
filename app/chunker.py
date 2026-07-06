from langchain_text_splitters import (RecursiveCharacterTextSplitter, 
CharacterTextSplitter, TokenTextSplitter, PythonCodeTextSplitter)


def get_splitter(
    splitter_name: str,
    chunk_size: int = 1000,
    overlap: int = 100,
):
    splitters = {
        "character": CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name = "cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "recursive": RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "token": TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "recursive_token": RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name = "gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
    }

    if splitter_name not in splitters:
        raise ValueError(
            f"Unsupported splitter: {splitter_name}"
        )

    return splitters[splitter_name]

def split_text(
    text: str,
    splitter_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
):
    splitter = get_splitter(
        splitter_name=splitter_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return splitter.split_text(text)