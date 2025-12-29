"""Application service that bridges FastAPI transport and the InterviewAgent."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from agents.interview_agent import InterviewAgent
from database import get_db, init_db
from schemas.interview import StartInterviewRequest, SubmitAnswerRequest
from skillsync.interviews.models import (
    InterviewAnswer,
    InterviewQuestion,
    InterviewSession,
    InterviewStatus,
)

# Ensure interview tables exist before the application starts serving requests
init_db()


def _status_value(status: InterviewStatus | str) -> str:
    return status.value if isinstance(status, InterviewStatus) else str(status)


class InterviewService:
    """Coordinates persistence logic for interview sessions."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.agent = InterviewAgent()

    async def start_session(self, payload: StartInterviewRequest) -> Dict[str, object]:
        return await self.agent.start_session(self.db, payload)

    async def submit_answer(self, payload: SubmitAnswerRequest) -> Dict[str, object]:
        return await self.agent.submit_answer(self.db, payload)

    async def get_report(self, interview_id: str) -> Dict[str, object]:
        return await self.agent.get_report(self.db, interview_id)

    async def get_next_question(self, interview_id: str) -> Dict[str, object]:
        """Get the next unanswered question in the interview."""
        session = self.db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        # Find next unanswered question
        next_question = (
            self.db.query(InterviewQuestion)
            .outerjoin(InterviewAnswer, InterviewQuestion.id == InterviewAnswer.question_id)
            .filter(InterviewQuestion.session_id == interview_id)
            .filter(InterviewAnswer.id.is_(None))
            .order_by(InterviewQuestion.order)
            .first()
        )
        
        if not next_question:
            return {
                "interview_id": interview_id,
                "status": "completed",
                "message": "No more questions available",
                "next_question": None
            }
        
        return {
            "interview_id": interview_id,
            "status": "active",
            "next_question": self._serialize_question(next_question),
            "progress": {
                "current": next_question.order,
                "total": len(session.questions or [])
            }
        }

    async def finish_interview(self, interview_id: str) -> Dict[str, object]:
        """Mark interview as completed and trigger report generation."""
        session = self.db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        session.status = InterviewStatus.COMPLETED
        self.db.commit()
        
        # Trigger report generation
        report = await self.agent.get_report(self.db, interview_id)
        
        return {
            "interview_id": interview_id,
            "status": "completed",
            "message": "Interview completed successfully",
            "report_ready": True,
            "report": report
        }

    def get_session(self, interview_id: str) -> Dict[str, object]:
        session = (
            self.db.query(InterviewSession)
            .options(
                joinedload(InterviewSession.questions).joinedload(InterviewQuestion.answer),
                joinedload(InterviewSession.report),
            )
            .filter(InterviewSession.id == interview_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")
        return self._serialize_session(session, include_questions=True)

    def list_sessions(self, *, user_id: Optional[str], limit: int) -> List[Dict[str, object]]:
        query = (
            self.db.query(InterviewSession)
            .options(joinedload(InterviewSession.questions))
            .order_by(InterviewSession.created_at.desc())
        )
        if user_id:
            query = query.filter(InterviewSession.user_id == user_id)
        sessions = query.limit(limit).all()
        return [self._serialize_session(session, include_questions=False) for session in sessions]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_session(self, session: InterviewSession, *, include_questions: bool) -> Dict[str, object]:
        questions = session.questions or []
        answered = sum(1 for q in questions if q.answer and q.answer.answer_text)

        payload: Dict[str, object] = {
            "interview_id": session.id,
            "user_id": session.user_id,
            "job_title": session.job_title,
            "job_description": session.job_description,
            "difficulty": session.difficulty,
            "status": _status_value(session.status),
            "overall_score": session.overall_score,
            "question_count": len(questions),
            "answered_questions": answered,
            "skills": session.skills or [],
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
            "report_ready": bool(session.report),
        }

        if include_questions:
            payload["questions"] = [self._serialize_question(question) for question in questions]
            if session.report:
                payload["analysis"] = session.report.report_content

        return payload

    def _serialize_question(self, question: InterviewQuestion) -> Dict[str, object]:
        return {
            "question_id": question.id,
            "question_text": question.question_text,
            "category": question.category or "general",
            "order": question.order,
            "max_score": question.max_score,
            "topics": [topic.name for topic in question.topics] if question.topics else [],
            "answer_text": question.answer.answer_text if question.answer else question.answer_text,
            "score": question.answer.score if question.answer else None,
            "ai_feedback": question.answer.ai_feedback if question.answer else None,
        }


def get_interview_service(db: Session = Depends(get_db)) -> "InterviewService":
    return InterviewService(db)
