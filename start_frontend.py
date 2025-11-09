#!/usr/bin/env python3
"""
SkillSync Frontend Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the SkillSync frontend development server"""
    
    print("🌐 Starting SkillSync Frontend - AI-Powered Job Search Revolution")
    print("✨ React Development Server with Tailwind CSS")
    print("\n📁 Frontend Features:")
    print("• 📄 Interactive CV Upload & Analysis")
    print("• 🎨 Portfolio Gallery & Generation")
    print("• 📊 Real-time Dashboard & Analytics")
    print("• 💡 Smart Recommendations Interface")
    print("• 🔄 Experience Translation Tools")
    print("• 🎨 Modern UI with Tailwind CSS")
    print("\n🔗 Frontend will be available at: http://localhost:3000")
    print("📊 Connect to backend at: http://localhost:8000\n")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Error: Frontend directory not found!")
        print(f"Expected: {frontend_dir}")
        sys.exit(1)
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("💿 Installing frontend dependencies...")
        print("Running: npm install")
        
        result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ npm install failed: {result.stderr}")
            print("\n💡 Try running manually:")
            print(f"cd {frontend_dir}")
            print("npm install")
            sys.exit(1)
        
        print("✅ Dependencies installed successfully!")
    
    try:
        print("🚀 Starting React development server...")
        print("🗑️ Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the development server
        subprocess.run(
            ["npm", "start"],
            cwd=frontend_dir,
            check=True
        )
    
    except KeyboardInterrupt:
        print("\n👋 SkillSync frontend stopped. Thank you!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting frontend: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure Node.js 16+ is installed")
        print("2. Try deleting node_modules and package-lock.json")
        print("3. Run 'npm install' manually")
        print("4. Check for port conflicts (port 3000)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()