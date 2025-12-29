from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import relationship

from models import Base


class InterviewStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


question_topic_links = Table(
    "question_topic_links",
    Base.metadata,
    Column("question_id", ForeignKey("interview_questions.id"), primary_key=True),
    Column("topic_id", ForeignKey("interview_topics.id"), primary_key=True),
)


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=True)
    cv_id = Column(String, nullable=True)

    job_title = Column(String, nullable=False)
    job_description = Column(Text, nullable=True)

    difficulty = Column(String, default="medium", nullable=False)
    skills = Column(JSON, nullable=True)
    status = Column(SAEnum(InterviewStatus), default=InterviewStatus.ACTIVE, nullable=False)

    overall_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    questions = relationship(
        "InterviewQuestion",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.order",
    )
    report = relationship("InterviewReport", back_populates="session", uselist=False)


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("interview_sessions.id"), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    order = Column(Integer, default=1)
    max_score = Column(Integer, nullable=True)
    answer_text = Column(Text, nullable=True)

    session = relationship("InterviewSession", back_populates="questions")
    topics = relationship(
        "Topic",
        secondary=question_topic_links,
        back_populates="questions",
    )
    answer = relationship(
        "InterviewAnswer",
        back_populates="question",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Topic(Base):
    __tablename__ = "interview_topics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)

    questions = relationship(
        "InterviewQuestion",
        secondary=question_topic_links,
        back_populates="topics",
    )


class InterviewAnswer(Base):
    __tablename__ = "interview_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("interview_questions.id"), nullable=False, unique=True)
    answer_text = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    ai_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    question = relationship("InterviewQuestion", back_populates="answer")


class InterviewReport(Base):
    __tablename__ = "interview_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("interview_sessions.id"), unique=True, nullable=False)
    file_path = Column(String, nullable=True)
    report_content = Column(JSON, nullable=True)
    sent_to_email = Column(Integer, default=0)
    strengths = Column(JSON, nullable=True)
    areas_for_improvement = Column(JSON, nullable=True)

    session = relationship("InterviewSession", back_populates="report")


__all__ = [
    "InterviewStatus",
    "InterviewSession",
    "InterviewQuestion",
    "InterviewAnswer",
    "InterviewReport",
    "Topic",
]
