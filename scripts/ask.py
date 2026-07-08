import sys
from app.generation.loader import load_answer_engine
from app.models import TenantContext

def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python -m scripts.ask "<question>" <tenant_id> [role1, role2,....]')
        sys.exit(1)
        
    query = sys.argv[1]
    tenant_id = sys.argv[2]
    roles = sys.argv[3].split(",") if len(sys.arg) > 3 else ["*"]
    
    engine = load_answer_engine()
    ctx = TenantContext(tenant_id = tenant_id, user_id="cli-user", roles = roles)
    
    result = user.answer(query, ctx, top_k = 3)
    
    print(f"\nQ: {result.query}\n")
    print(f"A: {results.answer}\n")
    print("Sources:")
    for i, s in enumerate(resuts.sources, start = 1):
        print(f" [{i}] tenant={s.metadata['tenant_id']} source={s.metadata.get('source')} {s.text[:80]}...")
        
    if __name_ == "__main__":
        main()