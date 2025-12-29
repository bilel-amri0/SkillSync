@echo off
REM SkillSync API - Phase 1.5 Quick Start (FIXED)

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

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo [2/3] Installing minimal dependencies (fast)...
echo This installs only FastAPI, PDF/DOCX parsing, and Jinja2
cd backend
pip install -r requirements_minimal.txt
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)
echo ✓ Minimal dependencies installed
echo.

echo [3/3] Starting SkillSync API server...
echo.
echo ⚠️ NOTE: Running with minimal dependencies
echo   - PDF parsing: ✓ Available
echo   - DOCX parsing: ✓ Available  
echo   - OCR (scanned PDFs): ✗ Not available (install pdf2image, pytesseract)
echo   - NLP skill extraction: ✗ Not available (install spacy)
echo   - Portfolio generation: ✓ Available
echo.
echo API will be available at:
echo   - http://localhost:8000/
echo   - http://localhost:8000/api/docs (Swagger UI)
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py

pause
