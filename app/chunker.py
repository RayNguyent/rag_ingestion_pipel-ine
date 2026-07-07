from langchain_text_splitters import (RecursiveCharacterTextSplitter, 
CharacterTextSplitter, TokenTextSplitter, PythonCodeTextSplitter)


def get_splitter(
    splitter_name: str,
    chunk_size: int = 1000,
    overlap: int = 100,
):
    builders = {
        "character": lambda: CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "recursive": lambda: RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "token": lambda: TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
        "recursive_token": lambda: RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
    }

    if splitter_name not in builders:
        raise ValueError(
            f"Unsupported splitter: {splitter_name}"
        )

    # Build only the requested splitter. Previously this dict was built
    # eagerly, which meant every call constructed all four splitters
    # (including two that fetch tiktoken vocab files over the network)
    # regardless of which one was actually requested.
    return builders[splitter_name]()

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