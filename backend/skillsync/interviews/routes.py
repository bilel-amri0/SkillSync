from typing import Optional

from fastapi import APIRouter, Depends, Query

from schemas.interview import (
    StartInterviewRequest, 
    SubmitAnswerRequest,
    NextQuestionRequest,
    FinishInterviewRequest
)
from .service import InterviewService, get_interview_service

router = APIRouter(prefix="/api/v2/interviews", tags=["AI Interviews"])


@router.post("/start")
async def start_interview(payload: StartInterviewRequest, svc: InterviewService = Depends(get_interview_service)):
    """Generate questions and create a new interview session."""
    return await svc.start_session(payload)


@router.post("/answer")
async def submit_answer(payload: SubmitAnswerRequest, svc: InterviewService = Depends(get_interview_service)):
    """Persist a candidate answer and trigger analysis when finished."""
    return await svc.submit_answer(payload)


@router.post("/next-question")
async def get_next_question(payload: NextQuestionRequest, svc: InterviewService = Depends(get_interview_service)):
    """Get the next question in the interview session (text mode)."""
    return await svc.get_next_question(payload.interview_id)


@router.post("/finish")
async def finish_interview(payload: FinishInterviewRequest, svc: InterviewService = Depends(get_interview_service)):
    """Mark interview as completed and generate final report."""
    return await svc.finish_interview(payload.interview_id)


@router.get("/{interview_id}")
def get_interview(interview_id: str, svc: InterviewService = Depends(get_interview_service)):
    """Return a detailed interview session including all questions and answers."""
    return svc.get_session(interview_id)


@router.get("/{interview_id}/report")
async def get_interview_report(interview_id: str, svc: InterviewService = Depends(get_interview_service)):
    """Fetch (or lazily generate) the AI interview report."""
    return await svc.get_report(interview_id)


@router.get("/")
def list_interviews(
    user_id: Optional[str] = Query(None, description="Filter by user identifier"),
    limit: int = Query(20, ge=1, le=100),
    svc: InterviewService = Depends(get_interview_service),
):
    """List the most recent interview sessions with lightweight summaries."""
    return svc.list_sessions(user_id=user_id, limit=limit)
