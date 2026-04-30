from app.core.llm import chat_completion


HYDE_SYSTEM_PROMPT = (
    "You are helping a search system. Given a question, write a single short "
    "paragraph (2-4 sentences) that would plausibly appear in a document that "
    "answers it. Use concrete domain language. Do not refuse, hedge, or note "
    "uncertainty — write the hypothetical answer as if you knew it. The output "
    "is used only for embedding-based retrieval; it will never be shown to "
    "users."
)


async def rewrite_for_retrieval(question: str, request_id: str | None = None) -> str:
    hypothetical = await chat_completion(
        system=HYDE_SYSTEM_PROMPT,
        user=question,
        request_id=request_id,
        temperature=0.0,
        max_tokens=200,
    )
    return f"{question}\n\n{hypothetical}".strip()
