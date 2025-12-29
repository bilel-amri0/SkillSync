@echo off
REM SkillSync Complete Project Startup Script (Windows)
REM Starts both backend and frontend servers

echo ========================================================
echo    SkillSync - Complete Project Startup
echo ========================================================
echo.

REM Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check Node.js
node --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo Requirements check passed!
echo.
echo Starting Backend Server...
start "SkillSync Backend" cmd /k "cd backend && python start_server.py"

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend Server...
start "SkillSync Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo    SkillSync is starting...
echo ========================================================
echo.
echo Backend:  http://localhost:8001
echo Frontend: http://localhost:5173
echo Docs:     http://localhost:8001/api/docs
echo.
echo Two terminal windows will open:
echo   1. Backend (FastAPI) - Keep this running
echo   2. Frontend (React) - Keep this running
echo.
echo Press any key to open browser...
pause > nul

timeout /t 3 /nobreak > nul
start http://localhost:5173

echo.
echo Project is running! Close this window anytime.
echo To stop servers, close both terminal windows.
echo.
pause
