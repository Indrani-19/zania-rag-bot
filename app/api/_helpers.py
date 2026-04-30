from fastapi import HTTPException, UploadFile

from app.config import settings
from app.core.qa import Answer
from app.models.schemas import QAItem, QAItemVerbose, SourceCitation


_READ_CHUNK_BYTES = 1024 * 1024


async def read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: exceeds the "
                    f"{settings.max_upload_size_mb} MB cap"
                ),
            )
    return bytes(buf)


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
