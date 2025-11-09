# Test Backend Only
# Start backend:
python -c "
import uvicorn
import sys
sys.path.append('.')
from main_enhanced import app

print('🚀 Starting SkillSync Backend...')
print('📍 Backend URL: http://localhost:8000')
print('📚 API Documentation: http://localhost:8000/docs')
print('💡 Health Check: http://localhost:8000/health')
print()
uvicorn.run(app, host='0.0.0.0', port=8000)
"