from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionsPayload(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def accept_list_or_object(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"questions": data}
        return data

    @field_validator("questions")
    @classmethod
    def reject_blank_questions(cls, qs: list[str]) -> list[str]:
        cleaned = [q.strip() for q in qs]
        if any(not q for q in cleaned):
            raise ValueError("questions must not be empty or whitespace-only")
        return cleaned


class SourceCitation(BaseModel):
    page: int | None = None
    snippet: str


class QAItem(BaseModel):
    question: str
    answer: str


class QAItemVerbose(QAItem):
    sources: list[SourceCitation] = []
    retrieval_score: float | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    chunk_count: int
    estimated_cost_usd: float


class ErrorResponse(BaseModel):
    type: str
    title: str
    status: int
    detail: str | None = None
