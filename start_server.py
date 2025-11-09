#!/usr/bin/env python3
"""
SkillSync Server Startup Script
"""

import sys
import uvicorn
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.main import app
from backend.config import settings

def main():
    """Start the SkillSync server"""
    
    print("🚀 Starting SkillSync - AI-Powered Job Search Revolution")
    print(f"🎯 Server running at: http://{settings.HOST}:{settings.PORT}")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    print("\n🔥 Features Available:")
    print("• 📄 Multi-format CV Analysis (PDF/DOCX)")
    print("• 🤖 AI-Powered Skill Extraction & Matching")
    print("• 🎨 Automatic Portfolio Generation")
    print("• 🔄 Experience Translation & Reformulation")
    print("• 🎯 Personalized Career Recommendations")
    print("• 📊 Explainable AI Insights")
    print("• 📈 Interactive Dashboard")
    print("\nℹ️ Ready to revolutionize your job search!\n")
    
    try:
        uvicorn.run(
            "backend.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 SkillSync server stopped. Thank you for using SkillSync!")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()