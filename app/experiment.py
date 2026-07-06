from app.loader import load_txt 
from app.chunker import split_text


doc = load_txt("data/sample.txt")

splitters = [
    "character",
    "recursive",
    "token",
    "recursive_token",
]

sizes = [256, 512, 1024]

overlaps = [20, 50, 100]

for splitter in splitters:
    for size in sizes:
        for overlap in overlaps:

            chunks = split_text(
                text=doc.text,
                splitter_name=splitter,
                chunk_size=size,
                overlap=overlap,
            )

            print(
                f"splitter={splitter}, "
                f"size={size}, "
                f"overlap={overlap}, "
                f"chunks={len(chunks)}"
            )
