@echo off
REM SkillSync Frontend Setup Script

echo ========================================
echo SkillSync Frontend - Automated Setup
echo ========================================
echo.

cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced

echo [1/7] Creating Vite React TypeScript project...
call npm create vite@latest frontend -- --template react-ts
if errorlevel 1 (
    echo ERROR: Failed to create Vite project
    pause
    exit /b 1
)

echo.
echo [2/7] Installing dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install base dependencies
    pause
    exit /b 1
)

echo.
echo [3/7] Installing additional packages...
call npm install axios lucide-react framer-motion clsx
if errorlevel 1 (
    echo ERROR: Failed to install additional packages
    pause
    exit /b 1
)

echo.
echo [4/7] Installing Tailwind CSS...
call npm install -D tailwindcss postcss autoprefixer
call npx tailwindcss init -p
if errorlevel 1 (
    echo ERROR: Failed to install Tailwind
    pause
    exit /b 1
)

echo.
echo [5/7] Setting up project structure...
mkdir src\components 2>nul
mkdir src\services 2>nul

echo.
echo [6/7] Copying component files...
copy ..\frontend_code\*.tsx src\ >nul 2>&1
copy ..\frontend_code\*.ts src\services\ >nul 2>&1
copy ..\frontend_code\CVUploader.tsx src\components\ >nul 2>&1
copy ..\frontend_code\TemplateSelector.tsx src\components\ >nul 2>&1
copy ..\frontend_code\ColorSchemeSelector.tsx src\components\ >nul 2>&1
copy ..\frontend_code\LoadingSpinner.tsx src\components\ >nul 2>&1
copy ..\frontend_code\index.css src\ >nul 2>&1

echo.
echo [7/7] Updating Tailwind config...
(
echo /** @type {import('tailwindcss'^).Config} */
echo export default {
echo   content: [
echo     "./index.html",
echo     "./src/**/*.{js,ts,jsx,tsx}",
echo   ],
echo   theme: {
echo     extend: {},
echo   },
echo   plugins: [],
echo }
) > tailwind.config.js

echo.
echo ========================================
echo ✓ Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure backend is running (port 8000^)
echo 2. Run: npm run dev
echo 3. Open: http://localhost:5173
echo.
pause
