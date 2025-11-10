"""
Pydantic models for interview functionality
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class StartInterviewRequest(BaseModel):
    """Request to start a new interview session"""
    cv_text: str = Field(..., description="The user's CV content")
    job_description: str = Field(..., description="The target job description")
    num_questions: Optional[int] = Field(default=5, description="Number of interview questions to generate")


class Question(BaseModel):
    """A single interview question"""
    question_id: int
    question_text: str
    category: str  # e.g., "technical", "behavioral", "situational"


class StartInterviewResponse(BaseModel):
    """Response when starting an interview"""
    interview_id: str
    questions: List[Question]
    message: str = "Interview session started successfully"


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer to a question"""
    question_id: int
    answer_text: str


class SubmitAnswerResponse(BaseModel):
    """Response after submitting an answer"""
    message: str = "Answer submitted successfully"
    next_question: Optional[Question] = None
    is_complete: bool = False


class InterviewTranscriptItem(BaseModel):
    """A single Q&A item in the interview transcript"""
    question_id: int
    question_text: str
    answer_text: str
    category: str


class InterviewAnalysis(BaseModel):
    """Analysis of the interview performance"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall performance score (0-100)")
    summary: str = Field(..., description="Summary of the interview performance")
    strengths: List[str] = Field(..., description="List of candidate strengths")
    weaknesses: List[str] = Field(..., description="List of areas for improvement")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class InterviewReportResponse(BaseModel):
    """Complete interview report"""
    interview_id: str
    cv_text: str
    job_description: str
    transcript: List[InterviewTranscriptItem]
    analysis: InterviewAnalysis
    created_at: str
