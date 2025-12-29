@echo off
REM Quick Start Script for SkillSync API (Phase 1.5)

echo ============================================
echo SkillSync API - Phase 1.5 Quick Start
echo ============================================
echo.

REM Check if we're in the correct directory
if not exist "backend\main.py" (
    echo ERROR: Please run this script from the SkillSync_Enhanced root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo [2/4] Installing dependencies...
cd backend
pip install fastapi uvicorn python-multipart pydantic
if errorlevel 1 (
    echo WARNING: Some dependencies failed to install
    echo You may need to install manually: pip install -r requirements.txt
)
echo ✓ Core dependencies installed
echo.

echo [3/4] Starting SkillSync API server...
echo.
echo API will be available at:
echo   - http://localhost:8000/
echo   - http://localhost:8000/api/docs (Swagger UI)
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py

pause
