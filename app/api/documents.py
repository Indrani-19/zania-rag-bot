import logging

from fastapi import APIRouter, File, Query, UploadFile

from app.api._helpers import read_capped, serialize_answers
from app.config import settings
from app.core import qa as qa_module
from app.core import vectorstore
from app.core.hashing import hash_file_bytes
from app.core.ingestion import ingest
from app.models.schemas import DocumentUploadResponse, QuestionsPayload
from app.utils.cost import tracker


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(document: UploadFile = File(...)) -> DocumentUploadResponse:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await read_capped(document, max_bytes)

    chunks = ingest(content, document.filename or "untitled")

    document_id = hash_file_bytes(content)
    cost_before = tracker.cumulative_cost_usd
    chunk_count = await vectorstore.index_document(
        document_id, chunks, request_id=document_id
    )

    return DocumentUploadResponse(
        document_id=document_id,
        chunk_count=chunk_count,
        estimated_cost_usd=round(tracker.cumulative_cost_usd - cost_before, 6),
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, int | str]:
    deleted = await vectorstore.delete_document(document_id)
    return {"document_id": document_id, "deleted_chunks": deleted}


@router.post("/documents/{document_id}/questions")
async def ask_questions(
    document_id: str,
    payload: QuestionsPayload,
    verbose: bool = Query(False),
):
    answers = await qa_module.answer_questions(
        document_id, payload.questions, request_id=document_id
    )
    return serialize_answers(answers, verbose)
