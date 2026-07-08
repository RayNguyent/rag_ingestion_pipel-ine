from typing import Any
from app.generation.bot_llm import LLMClient
from app.generation.prompt import build_rag_prompt
from app.models import AnswerResult, TenantContext
from app.retrieval.engine import ContextRetrievalEngine

class RAGAnswerEngine:
    """
    The full entrypoint: retrieval -> prompt construction -> LLM call -> cited answer.
    """
    
    def __init__(self, retrieval_engine: ContextRetrievalEngine, llm_client: LLMClient):
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        
    def answer(
        query: str,
        tenant_context: TenantContext,
        filters: dict[str, Any]| None =  None,
        top_k: int | None = None
    ) -> AnswerResult :
        results = self.retrieval_engine.retrieve(query, tenant_context, filters=filters, top_k = top_k)
        
        if not results:
            return AnswerResult(
                query=query,
                answer="I dont have any info about that in the docs",
                sources = []
            )
        system_prompt, user_prompt = build_rag_prompt(query, results)
        answer_text = self.llm_client.generate(system_prompt, user_prompt)
        
        return AnswerResult(query=query, answer=answer_text, sources = results)