#!/usr/bin/env python3
"""
SkillSync Backend Server Startup Script
Starts the FastAPI backend server with all features enabled
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the SkillSync backend server"""
    
    print("=" * 60)
    print(" Starting SkillSync Backend Server")
    print("=" * 60)
    print()
    print(" Server will be available at:")
    print("   http://localhost:8001")
    print("   http://127.0.0.1:8001")
    print()
    print(" API Documentation:")
    print("   http://localhost:8001/api/docs")
    print("   http://localhost:8001/api/redoc")
    print()
    print(" Features Enabled:")
    print("    CV Analysis & Skill Extraction")
    print("    Multi-API Job Search (Adzuna, The Muse, RemoteOK)")
    print("    AI-Powered Recommendations")
    print("    Portfolio Generator")
    print("    Experience Translator")
    print("    AI Interview System (Text & Voice)")
    print("    XAI Explainable AI")
    print("    JWT Authentication")
    print()
    print(" Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("  Warning: .env file not found")
        print("   Creating from .env.example...")
        env_example = Path(__file__).parent / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("    .env file created")
        else:
            print("     Using default configuration")
        print()
    
    try:
        # Start server using main_simple_for_frontend (main application)
        uvicorn.run(
            "main_simple_for_frontend:app",
            host="0.0.0.0",
            port=8001,
            reload=False,  # DISABLED: Reload causes cache issues
            log_level="info",
            access_log=True,
            workers=1  # Single worker for development
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  Server stopped by user")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure port 8001 is not already in use")
        print("2. Check if all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("3. Try running directly:")
        print("   python -m uvicorn main_simple_for_frontend:app --port 8001")
        sys.exit(1)

if __name__ == "__main__":
    main()
