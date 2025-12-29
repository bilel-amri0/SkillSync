@echo off
REM ========================================
REM   COMPLETE TESTING COMMANDS
REM ========================================

echo.
echo ========================================
echo   STEP 1: START SERVER
echo ========================================
echo.
echo Run this command in Terminal 1:
echo.
echo   cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
echo   python -m uvicorn main_simple_for_frontend:app --reload --port 8001
echo.
echo Wait until you see: "Application startup complete"
echo.
pause

echo.
echo ========================================
echo   STEP 2: TEST BOTH ENDPOINTS
echo ========================================
echo.
echo Run this command in Terminal 2:
echo.
echo   cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
echo   python test_both_endpoints.py
echo.
echo This will compare standard vs advanced ML endpoints
echo.
pause

echo.
echo ========================================
echo   STEP 3: TEST PRODUCTION PARSER
echo ========================================
echo.
echo Run this command in Terminal 2:
echo.
echo   cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
echo   python final_validation.py
echo.
echo This validates the upgraded production parser
echo.
pause

echo.
echo ========================================
echo   STEP 4: TEST ADVANCED PARSER
echo ========================================
echo.
echo Run this command in Terminal 2:
echo.
echo   cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
echo   python -c "from advanced_cv_parser import AdvancedCVParser; parser = AdvancedCVParser(); result = parser.parse_cv('Senior engineer with Python, AWS. Led team of 5.'); print(f'Skills: {len(result.skills)}, Industries: {len(result.industries)}, Seniority: {result.seniority_level}')"
echo.
pause

echo.
echo ========================================
echo   ALL TESTS COMPLETE!
echo ========================================
echo.
