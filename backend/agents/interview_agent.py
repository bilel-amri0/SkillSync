"""High-level orchestrator for SkillSync's AI-powered interviews."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from integrations.email_service import EmailService
from integrations.gemini_client import GeminiClient
from schemas.interview import StartInterviewRequest, SubmitAnswerRequest
from skillsync.interviews.models import (
    InterviewAnswer,
    InterviewQuestion,
    InterviewReport,
    InterviewSession,
    InterviewStatus,
)

logger = logging.getLogger(__name__)


class InterviewAgent:
    """Coordinates question generation, persistence, evaluation, and reporting."""

    def __init__(self) -> None:
        self.gemini_client = GeminiClient()
        self.email_service = EmailService()

    async def start_session(self, db: Session, request: StartInterviewRequest) -> Dict[str, object]:
        """Validate input, generate AI questions, and persist an interview session."""
        questions_data = await self.gemini_client.generate_questions(
            cv_text=request.cv_text,
            job_description=request.job_description or "",
            job_title=request.job_title,
            difficulty=request.difficulty,
            skills=request.skills,
        )

        if not questions_data:
            raise HTTPException(status_code=500, detail="AI failed to generate questions")

        session_id = str(uuid.uuid4())
        db_session = InterviewSession(
            id=session_id,
            user_id=request.user_id,
            cv_id=request.cv_id,
            job_title=request.job_title,
            job_description=request.job_description,
            difficulty=request.difficulty,
            skills=request.skills,
            status=InterviewStatus.ACTIVE,
        )

        for idx, raw_question in enumerate(questions_data, start=1):
            question_text = raw_question
            if isinstance(raw_question, dict):
                question_text = raw_question.get("question_text") or raw_question.get("question") or "Tell me about yourself."

            db_session.questions.append(
                InterviewQuestion(
                    question_text=question_text.strip(),
                    category=(raw_question.get("category") if isinstance(raw_question, dict) else None),
                    order=idx,
                )
            )

        db.add(db_session)
        db.commit()
        db.refresh(db_session)

        logger.info("Created interview session %s with %s questions", session_id, len(db_session.questions))

        return {
            "interview_id": session_id,
            "questions": [
                {
                    "question_id": question.id,
                    "question_text": question.question_text,
                    "category": question.category or "general",
                    "order": question.order,
                }
                for question in db_session.questions
            ],
        }

    async def submit_answer(self, db: Session, request: SubmitAnswerRequest) -> Dict[str, object]:
        """Persist the answer and signal progress."""
        db_question = (
            db.query(InterviewQuestion)
            .options(joinedload(InterviewQuestion.session))
            .filter(
                InterviewQuestion.id == request.question_id,
                InterviewQuestion.session_id == request.interview_id,
            )
            .first()
        )

        if not db_question:
            raise HTTPException(status_code=404, detail="Question not found")

        if db_question.answer:
            db_question.answer.answer_text = request.answer_text
            db_question.answer.created_at = datetime.utcnow()
        else:
            db_question.answer = InterviewAnswer(answer_text=request.answer_text)

        db.commit()

        session = db_question.session
        total_questions = len(session.questions)
        answered_questions = sum(1 for q in session.questions if q.answer and q.answer.answer_text)

        logger.debug("Interview %s progress %s/%s", session.id, answered_questions, total_questions)

        if answered_questions == total_questions:
            await self._analyze_session(db, session)

        return {"status": "success", "progress": f"{answered_questions}/{total_questions}"}

    async def get_report(self, db: Session, interview_id: str) -> Dict[str, object]:
        """Return a full transcript and AI analysis for an interview."""
        session = (
            db.query(InterviewSession)
            .options(
                joinedload(InterviewSession.questions).joinedload(InterviewQuestion.answer),
                joinedload(InterviewSession.report),
            )
            .filter(InterviewSession.id == interview_id)
            .first()
        )

        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")

        if not session.overall_score:
            await self._analyze_session(db, session)

        transcript = [
            {
                "question": question.question_text,
                "answer": (question.answer.answer_text if question.answer else question.answer_text),
                "order": question.order,
            }
            for question in session.questions
        ]

        analysis_payload = session.report.report_content if session.report else {}

        return {
            "interview_id": session.id,
            "job_title": session.job_title,
            "created_at": session.created_at,
            "status": session.status.value if hasattr(session.status, "value") else session.status,
            "transcript": transcript,
            "analysis": analysis_payload,
        }

    async def _analyze_session(self, db: Session, session: InterviewSession) -> None:
        """Call Gemini for grading/reporting and persist the results."""
        transcript_payload = [
            {
                "question": question.question_text,
                "answer": (question.answer.answer_text if question.answer else question.answer_text or ""),
                "order": question.order,
            }
            for question in session.questions
        ]

        analysis = await self.gemini_client.analyze_transcript(
            job_title=session.job_title,
            transcript=transcript_payload,
        )

        session.overall_score = analysis.get("overall_score", 75.0)
        session.status = InterviewStatus.COMPLETED
        session.updated_at = datetime.utcnow()

        if session.report:
            session.report.report_content = analysis
            session.report.strengths = analysis.get("strengths")
            session.report.areas_for_improvement = analysis.get("weaknesses")
        else:
            session.report = InterviewReport(
                session_id=session.id,
                report_content=analysis,
                strengths=analysis.get("strengths"),
                areas_for_improvement=analysis.get("weaknesses"),
            )

        db.commit()

        recipient_value = analysis.get("recipient_email")
        recipient = str(recipient_value) if recipient_value else None
        if recipient:
            self.email_service.queue_report_email(
                to_email=recipient,
                subject=f"Interview results for {session.job_title}",
                report_payload=analysis,
            )

        logger.info("Interview %s analyzed with score %.2f", session.id, session.overall_score)