from app.generation.bot_llm import get_llm_client
from app.generation.pipeline import RAGAnswerEngine
from app.retrieval.loader import load_engine

def load_answer_engine() -> RAGAnswerEngine:
    retrieval_engine = load_engine()
    llm_client = get_llm_client()
    return RAGAnswerEngine(retrieval_engine, llm_client)