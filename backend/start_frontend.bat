@echo off
REM SkillSync Frontend Startup Script (Windows)
echo ========================================================
echo    Starting SkillSync Frontend
echo ========================================================
echo.

cd ..\frontend

if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    echo.
)

echo Starting development server...
echo.
echo Frontend will be available at:
echo    http://localhost:5173
echo    http://127.0.0.1:5173
echo.
echo Press Ctrl+C to stop the server
echo ========================================================
echo.

call npm run dev
