#!/bin/bash

# SkillSync XAI System Setup Script
# This script sets up the complete XAI (Explainable AI) system for SkillSync

set -e  # Exit on any error

echo "🚀 SkillSync XAI System Setup"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the SkillSync project root."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies including XAI libraries
echo "📚 Installing backend dependencies..."
pip install -r requirements.txt

# Install additional XAI-specific dependencies
echo "🤖 Installing XAI-specific dependencies..."
pip install shap==0.43.0
pip install lime==0.2.0.1
pip install matplotlib seaborn plotly

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Check if XAI libraries are available
echo ""
echo "🔍 Checking XAI library installation..."
python3 -c "
import sys
try:
    import shap
    print('✅ SHAP installed successfully')
except ImportError:
    print('❌ SHAP not available')
    sys.exit(1)

try:
    import lime
    print('✅ LIME installed successfully')
except ImportError:
    print('❌ LIME not available')
    sys.exit(1)

try:
    import matplotlib
    print('✅ Matplotlib installed successfully')
except ImportError:
    print('❌ Matplotlib not available')
    sys.exit(1)

print('🎉 All XAI libraries installed successfully!')
"

if [ $? -ne 0 ]; then
    echo "❌ XAI library installation check failed"
    exit 1
fi

echo ""
echo "🧪 Running XAI System Tests..."
python3 test_xai_system.py

if [ $? -eq 0 ]; then
    echo "✅ XAI System Tests PASSED"
else
    echo "❌ XAI System Tests FAILED"
    echo "⚠️ Setup completed but tests failed. Check the error messages above."
fi

echo ""
echo "🎯 XAI System Setup Complete!"
echo "================================"
echo ""
echo "📋 What was installed:"
echo "   • SHAP (SHapley Additive exPlanations) for feature importance"
echo "   • LIME (Local Interpretable Model-agnostic Explanations) for local explanations"
echo "   • Matplotlib/Plotly for visualization"
echo "   • Enhanced XAI API endpoints"
echo "   • Frontend XAI Dashboard components"
echo ""
echo "🚀 To start using the XAI system:"
echo "   1. Start the backend: python3 backend/main_simple_for_frontend.py"
echo "   2. Start the frontend: cd frontend && npm start"
echo "   3. Upload a CV and check the 'AI Explanations' tab"
echo ""
echo "📊 XAI Features Available:"
echo "   • Transparent skill extraction explanations"
echo "   • SHAP-based job matching feature importance"
echo "   • LIME text analysis for CV content"
echo "   • Interactive XAI Dashboard"
echo "   • 80% explainability compliance tracking"
echo ""
echo "📖 For detailed usage instructions, see:"
echo "   • XAI implementation documentation"
echo "   • test_xai_system.py for testing examples"
echo "   • frontend/src/components/XAIDashboard.js for UI integration"
echo ""

# Create a simple startup script
cat > start_xai_system.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting SkillSync XAI System..."

# Start backend in background
echo "Starting backend API..."
python3 backend/main_simple_for_frontend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd frontend && npm start &
FRONTEND_PID=$!

echo ""
echo "✅ XAI System started!"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "🌐 Access the application at: http://localhost:3000"
echo "🔍 Check XAI explanations in the 'AI Explanations' tab"
echo ""
echo "To stop the system, run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "echo ''; echo '🛑 Stopping XAI System...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
EOF

chmod +x start_xai_system.sh

echo "📝 Created start_xai_system.sh script for easy system startup"
echo ""
echo "🎉 Setup completed successfully!"
echo "Run './start_xai_system.sh' to start the complete XAI-enabled system."