from app.models import SearchResult

SYSTEM_PROMPT = (
    "You are a retrieval-augmented generation assistant. Answer  the user's question using ONLY the numbered context passages provided below -"
    "do not use outside knowledge or make up answers. If the answer is not contained within the context, say ""I don't know."" After each claim, cite the passage numbers it"
    "came from, like [1] or [1][3]" 
)

def build_rag_prompt(
    query: str,
    results: list[SearchResult],
    max_context_chars: int | None = None
) -> tuple[str,str]:
    """
    Build a RAG prompt for the LLM, given a query and a list of SearchResult objects.

    Returns a tuple of (system_prompt, user_prompt).
    """
    from app.config import settings
    budget = max_context_chars or settings.max_context_chars
    
    blocks = []
    used = 0
    for i, result in enumerate(results,start = 1):
        source = result.metadata.get("source", "unknown source")
        block = f"[{i}] {result.text} (source: {source})\n"
        if used + len(block) >  budget and blocks:
            break
        blocks.append(block)
        used += len(block)
        
    context_str = "\n\n".join(blocks) if blocks else "No context available."
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    return SYSTEM_PROMPT, user_prompt