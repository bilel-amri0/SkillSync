@echo off
echo ========================================
echo   SKILLSYNC - ADVANCED ML CV PARSER
echo ========================================
echo.
echo Starting server with Advanced ML integration...
echo.
echo Two endpoints available:
echo   1. /api/v1/analyze-cv (standard)
echo   2. /api/v1/analyze-cv-advanced (ML-powered)
echo.
echo Server will start on: http://localhost:8001
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
python -m uvicorn main_simple_for_frontend:app --reload --port 8001
