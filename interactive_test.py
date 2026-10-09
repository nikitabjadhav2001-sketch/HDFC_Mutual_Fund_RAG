import asyncio
from app.orchestrator import answer_question

async def main():
    print("=== Interactive RAG Test (Groq + Postgres) ===")
    print("Type your question below (or 'exit' to quit):\n")
    while True:
        try:
            query = input("Q: ").strip()
            if not query or query.lower() in ["exit", "quit"]:
                break
            res = await answer_question(query)
            print(f"\nA: {res.answer}\n")
            if hasattr(res, "sources") and res.sources:
                print("Sources:", [s.doc_title for s in res.sources])
            print("-" * 50)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())