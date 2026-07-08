from app.generation.llm import StubLLMClient
from app.generation.pipeline import RAGAnswerEngine
from app.generation.prompt import build_rag_prompt
from app.models import SearchResult, TenantContext


def _result(chunk_id, text, source="doc.txt", tenant_id="tenant-a"):
    return SearchResult(
        chunk_id=chunk_id,
        document_id="d1",
        text=text,
        metadata={"source": source, "tenant_id": tenant_id, "acl_roles": ["*"]},
    )


def test_prompt_includes_all_context_and_citation_instructions():
    results = [_result("a", "first passage"), _result("b", "second passage")]
    system, user = build_rag_prompt("my question", results)

    assert "cite" in system.lower()
    assert "[1]" in user and "[2]" in user
    assert "first passage" in user
    assert "second passage" in user
    assert "my question" in user


def test_prompt_respects_context_char_budget():
    results = [_result("a", "x" * 100), _result("b", "y" * 100)]
    _, user = build_rag_prompt("q", results, max_context_chars=50)

    # first block always included even if it alone exceeds budget; second dropped
    assert "x" * 100 in user
    assert "y" * 100 not in user


def test_prompt_handles_no_results():
    system, user = build_rag_prompt("q", [])
    assert "no context retrieved" in user


def test_stub_llm_client_returns_top_passage_labeled_as_stub():
    results = [_result("a", "the answer is 42")]
    _, user_prompt = build_rag_prompt("what is the answer", results)

    answer = StubLLMClient().generate("system", user_prompt)

    assert "the answer is 42" in answer
    assert "StubLLMClient" in answer


def test_answer_engine_returns_grounded_answer_with_sources(two_tenant_engine):
    from app.retrieval.engine import ContextRetrievalEngine

    retrieval_engine, _, _ = two_tenant_engine
    answer_engine = RAGAnswerEngine(retrieval_engine, StubLLMClient())
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])

    result = answer_engine.answer("cloud infrastructure margin", ctx, top_k=3)

    assert result.query == "cloud infrastructure margin"
    assert len(result.sources) > 0
    assert all(s.metadata["tenant_id"] == "tenant-a" for s in result.sources)
    assert result.answer  # non-empty


def test_answer_engine_handles_zero_results_gracefully(two_tenant_engine):
    retrieval_engine, _, _ = two_tenant_engine
    answer_engine = RAGAnswerEngine(retrieval_engine, StubLLMClient())
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])

    result = answer_engine.answer("completely unrelated gibberish xyzzy", ctx, top_k=3)

    # even with weak matches, engine should not error; if truly empty, it
    # should return the graceful fallback message and no sources
    assert isinstance(result.answer, str)
    if not result.sources:
        assert "don't have any information" in result.answer
