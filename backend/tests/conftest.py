"""Pytest fixtures for backend tests with isolated database."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from database import get_db
from models import Base

TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///./test_skillsync.db")
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False} if TEST_DB_URL.startswith("sqlite") else {},
)
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _override_get_db() -> Generator:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def reset_database() -> None:
    """Ensure a clean schema before every test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient wired to the test database."""
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
