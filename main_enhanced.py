"""Root-level ASGI entrypoint to support `uvicorn main_enhanced:app` from repo top."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
	sys.path.append(str(BACKEND_PATH))

from backend.main_enhanced import app  # noqa: F401  pylint: disable=wrong-import-position
