"""Pydantic contracts shared by interview endpoints."""

from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class StartInterviewRequest(BaseModel):
    user_id: str = Field(..., description="Identifier for the candidate/user initiating the session")
    cv_id: Optional[str] = Field(None, description="Optional identifier of the uploaded CV")
    cv_text: str = Field(..., description="Plain-text content extracted from the CV")
    job_title: str
    job_description: Optional[str] = None
    difficulty: str = Field("medium", description="Difficulty level (easy/medium/hard)")
    skills: List[str] = Field(default_factory=list)
    interview_mode: Literal["text", "voice"] = Field("text", description="Interview mode: text-based or live voice")


class SubmitAnswerRequest(BaseModel):
    interview_id: str
    question_id: int
    answer_text: str


class NextQuestionRequest(BaseModel):
    interview_id: str = Field(..., description="Current interview session ID")


class FinishInterviewRequest(BaseModel):
    interview_id: str = Field(..., description="Interview session ID to finish")


class InterviewQuestionOut(BaseModel):
    question_id: int
    question_text: str
    category: str = "general"
    order: int


class InterviewOut(BaseModel):
    interview_id: str
    user_id: str
    job_title: str
    interview_mode: str
    status: str
    current_question: Optional[InterviewQuestionOut] = None
    total_questions: int = 0
    answered_questions: int = 0
