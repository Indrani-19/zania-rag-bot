import json
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api._helpers import read_capped, serialize_answers
from app.config import settings
from app.core import qa as qa_module
from app.core import vectorstore
from app.core.hashing import hash_file_bytes
from app.core.ingestion import ingest
from app.models.schemas import QuestionsPayload


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/qa")
async def upload_and_ask(
    document: UploadFile = File(...),
    questions: UploadFile = File(...),
    verbose: bool = Query(False),
):
    request_id = str(uuid.uuid4())

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    doc_content = await read_capped(document, max_bytes)
    chunks = ingest(doc_content, document.filename or "untitled")

    questions_raw = await questions.read()
    try:
        questions_data = json.loads(questions_raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid questions JSON: {e}"
        ) from e

    payload = QuestionsPayload.model_validate(questions_data)

    document_id = hash_file_bytes(doc_content)
    await vectorstore.index_document(document_id, chunks, request_id=request_id)
    answers = await qa_module.answer_questions(
        document_id, payload.questions, request_id=request_id
    )
    return serialize_answers(answers, verbose)
