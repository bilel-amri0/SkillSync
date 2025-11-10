"""
FastAPI router for interview endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from models.interview_models import (
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    InterviewReportResponse,
    Question,
    InterviewAnalysis,
)
from agents.interview_agent import get_interview_agent


router = APIRouter(
    prefix="/api/v1/interviews",
    tags=["interviews"],
)


@router.post("/start", response_model=StartInterviewResponse, status_code=status.HTTP_201_CREATED)
async def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    """
    Start a new interview session
    
    Generates tailored interview questions based on the provided CV and job description.
    
    Args:
        request: StartInterviewRequest with cv_text, job_description, and optional num_questions
    
    Returns:
        StartInterviewResponse with interview_id and generated questions
    """
    try:
        agent = get_interview_agent()
        result = agent.start_interview(
            cv_text=request.cv_text,
            job_description=request.job_description,
            num_questions=request.num_questions or 5
        )
        
        questions = [Question(**q) for q in result["questions"]]
        
        return StartInterviewResponse(
            interview_id=result["interview_id"],
            questions=questions
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview: {str(e)}"
        )


@router.post("/interviews/{interview_id}/submit_answer", response_model=SubmitAnswerResponse)
async def submit_answer(interview_id: str, request: SubmitAnswerRequest) -> SubmitAnswerResponse:
    """
    Submit an answer to a question
    
    Args:
        interview_id: The interview session ID
        request: SubmitAnswerRequest with question_id and answer_text
    
    Returns:
        SubmitAnswerResponse with submission status and next question if available
    """
    try:
        agent = get_interview_agent()
        result = agent.submit_answer(
            interview_id=interview_id,
            question_id=request.question_id,
            answer_text=request.answer_text
        )
        
        next_question = None
        if result["next_question"]:
            next_question = Question(**result["next_question"])
        
        return SubmitAnswerResponse(
            next_question=next_question,
            is_complete=result["is_complete"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit answer: {str(e)}"
        )


@router.get("/interviews/{interview_id}/report", response_model=InterviewReportResponse)
async def get_interview_report(interview_id: str) -> InterviewReportResponse:
    """
    Get the complete interview report
    
    Retrieves the full interview transcript and AI-generated analysis.
    
    Args:
        interview_id: The interview session ID
    
    Returns:
        InterviewReportResponse with complete report including analysis
    """
    try:
        agent = get_interview_agent()
        report = agent.get_report(interview_id)
        
        return InterviewReportResponse(
            interview_id=report["interview_id"],
            cv_text=report["cv_text"],
            job_description=report["job_description"],
            transcript=report["transcript"],
            analysis=InterviewAnalysis(**report["analysis"]),
            created_at=report["created_at"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get interview report: {str(e)}"
        )
