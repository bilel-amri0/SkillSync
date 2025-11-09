#!/bin/bash

# SkillSync Portfolio Generator Test Script
# This script helps you quickly test the CV analysis and portfolio generation

echo "🚀 SkillSync Portfolio Generator - Quick Test"
echo "=============================================="

# Check if backend and frontend are running
echo ""
echo "1. Starting Backend..."
cd backend
if [ ! -f "main.py" ]; then
    echo "❌ Backend not found. Please make sure you're in the correct directory."
    exit 1
fi

# Start backend in background
python main.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 3

echo ""
echo "2. Starting Frontend..."
cd ../frontend

# Start frontend in background
npm start &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "🎉 SkillSync is now running!"
echo ""
echo "📋 Instructions:"
echo "1. Open your browser to http://localhost:3000"
echo "2. Go to CV Analysis page"
echo "3. Upload the test CV file: test_cv_bilel_amri.txt"
echo "4. Wait for analysis to complete"
echo "5. Go to Portfolio page to generate your portfolio"
echo ""
echo "📁 Test CV file location: ../test_cv_bilel_amri.txt"
echo ""
echo "🛑 To stop the servers, press Ctrl+C and run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Wait for user to press Ctrl+C
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; echo '✅ Servers stopped.'; exit 0" INT

echo "Press Ctrl+C to stop the servers..."
wait