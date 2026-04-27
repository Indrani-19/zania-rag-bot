from fastapi import HTTPException

from app.config import settings
from app.core.qa import Answer
from app.models.schemas import QAItem, QAItemVerbose, SourceCitation


def check_size(content: bytes) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(content) / 1024 / 1024:.1f} MB exceeds the "
                f"{settings.max_upload_size_mb} MB cap"
            ),
        )


def serialize_answers(
    answers: list[Answer], verbose: bool
) -> list[QAItem] | list[QAItemVerbose]:
    if verbose:
        return [
            QAItemVerbose(
                question=a.question,
                answer=a.answer,
                sources=[SourceCitation(page=s.page, snippet=s.snippet) for s in a.sources],
                retrieval_score=a.retrieval_score,
            )
            for a in answers
        ]
    return [QAItem(question=a.question, answer=a.answer) for a in answers]
