import json
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api._helpers import check_size, serialize_answers
from app.core import qa as qa_module
from app.core import vectorstore
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

    doc_content = await document.read()
    check_size(doc_content)
    chunks = ingest(doc_content, document.filename or "untitled")

    questions_raw = await questions.read()
    try:
        questions_data = json.loads(questions_raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid questions JSON: {e}"
        ) from e

    payload = QuestionsPayload.model_validate(questions_data)

    document_id = str(uuid.uuid4())
    await vectorstore.index_document(document_id, chunks, request_id=request_id)
    answers = await qa_module.answer_questions(
        document_id, payload.questions, request_id=request_id
    )
    return serialize_answers(answers, verbose)
