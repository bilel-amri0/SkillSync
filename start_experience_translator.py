#!/usr/bin/env python3
"""
Experience Translator (F7) Startup and Demo Script
Starts both backend and frontend services and provides demo functionality
"""

import subprocess
import sys
import os
import time
import requests
import threading
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("🚀 SkillSync Experience Translator (F7)")
    print("Advanced NLG-powered Experience Reformulation System")
    print("=" * 60)
    print(f"{Colors.ENDC}")

def print_step(step_num, title):
    print(f"{Colors.BOLD}[Step {step_num}] {title}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")

def check_requirements():
    """Check if required files and directories exist"""
    print_step(1, "Checking Requirements")
    
    required_files = [
        'backend/main_simple_for_frontend.py',
        'backend/experience_translator.py',
        'frontend/package.json',
        'frontend/src/pages/ExperienceTranslator.js'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_error("Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print_success("All required files found")
    return True

def start_backend():
    """Start the backend server"""
    print_step(2, "Starting Backend Server")
    
    backend_dir = Path("backend")
    backend_file = backend_dir / "main_simple_for_frontend.py"
    
    try:
        # Change to backend directory and start server
        os.chdir(backend_dir)
        
        print_info("Starting FastAPI backend server...")
        print_info("Backend will be available at: http://localhost:8001")
        
        # Start the backend process
        process = subprocess.Popen([
            sys.executable, "main_simple_for_frontend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print_success("Backend server started (PID: {})".format(process.pid))
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if backend is responding
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print_success("Backend health check passed")
                return process
            else:
                print_error(f"Backend health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException:
            print_warning("Backend may still be starting up...")
            return process
            
    except Exception as e:
        print_error(f"Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print_step(3, "Starting Frontend Server")
    
    frontend_dir = Path("../frontend")
    
    try:
        os.chdir(frontend_dir)
        
        print_info("Starting React frontend development server...")
        print_info("Frontend will be available at: http://localhost:3000")
        
        # Start the frontend process
        process = subprocess.Popen([
            "npm", "start"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print_success("Frontend server started (PID: {})".format(process.pid))
        
        # Wait for frontend to start
        print_info("Waiting for frontend to initialize...")
        time.sleep(10)
        
        return process
        
    except Exception as e:
        print_error(f"Failed to start frontend: {e}")
        print_info("You can manually start frontend with: cd frontend && npm start")
        return None

def test_experience_translator():
    """Test the Experience Translator functionality"""
    print_step(4, "Testing Experience Translator")
    
    try:
        # Test data
        test_experience = """
        Worked on various web development projects using React and Node.js. 
        Built applications, collaborated with teams, and improved system performance. 
        Participated in code reviews and followed agile methodologies.
        """
        
        test_job_description = """
        Senior Full-Stack Developer position:
        
        Requirements:
        - 3+ years React.js, Node.js, TypeScript experience
        - Cloud platforms (AWS, Azure) knowledge
        - Microservices architecture experience
        - Performance optimization expertise
        - Team leadership and mentoring experience
        - DevOps and CI/CD pipeline knowledge
        
        Responsibilities:
        - Lead development of scalable web applications
        - Optimize application performance and reliability
        - Mentor junior developers and lead technical initiatives
        """
        
        # Test the translation
        print_info("Testing experience translation...")
        
        response = requests.post(
            "http://localhost:8001/api/v1/experience/translate",
            json={
                "original_experience": test_experience,
                "job_description": test_job_description,
                "style": "professional"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success("Experience translation test successful!")
            print(f"   - Translation ID: {result.get('translation_id', 'N/A')}")
            print(f"   - Confidence Score: {result.get('confidence_score', 0):.2f}")
            print(f"   - Rewriting Style: {result.get('rewriting_style', 'N/A')}")
            print(f"   - Keyword Matches: {len(result.get('keyword_matches', {}))}")
            print(f"   - Enhancements: {len(result.get('enhancements_made', []))}")
            print(f"   - Suggestions: {len(result.get('suggestions', []))}")
            
            return True
        else:
            print_error(f"Translation test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Translation test failed: {e}")
        return False

def show_demo_instructions():
    """Show instructions for using the demo"""
    print_step(5, "Experience Translator Demo")
    
    print(f"{Colors.BOLD}🎯 Ready to Use Experience Translator!{Colors.ENDC}")
    print()
    print_info("Access Points:")
    print("   📱 Frontend: http://localhost:3000")
    print("   🔧 Backend API: http://localhost:8001")
    print("   📚 API Docs: http://localhost:8001/docs")
    print()
    
    print(f"{Colors.BOLD}🚀 How to Use:{Colors.ENDC}")
    print("1. Open http://localhost:3000 in your browser")
    print("2. Navigate to 'Experience Translator' page")
    print("3. Paste your experience description")
    print("4. Paste target job description")
    print("5. Choose rewriting style (Professional/Technical/Creative)")
    print("6. Click 'Translate Experience'")
    print("7. Review enhanced version and export in your preferred format")
    print()
    
    print(f"{Colors.BOLD}💡 Test Examples:{Colors.ENDC}")
    print("Original Experience:")
    print('   "Worked on web projects using different technologies"')
    print()
    print("Job Description Keywords:")
    print('   "React.js, Node.js, TypeScript, AWS, leadership, performance"')
    print()
    print("Expected Enhancement:")
    print('   "• Developed scalable web applications using React.js and Node.js')
    print('    • Collaborated with cross-functional teams to deliver high-quality solutions')
    print('    • Optimized application performance through efficient coding practices"')
    print()

def main():
    """Main execution function"""
    print_banner()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check requirements
    if not check_requirements():
        print_error("Requirements check failed. Please ensure all files are present.")
        return 1
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print_error("Failed to start backend. Please check the error messages above.")
        return 1
    
    # Test Experience Translator
    if not test_experience_translator():
        print_warning("Experience Translator test failed, but servers are running")
    
    # Show demo instructions
    show_demo_instructions()
    
    # Keep the script running
    print(f"{Colors.BOLD}{Colors.GREEN}🎉 SkillSync Experience Translator is Ready!{Colors.ENDC}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.ENDC}")
    print()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        print_info("Shutting down services...")
        
        # Stop backend
        if backend_process:
            backend_process.terminate()
            print_info("Backend server stopped")
        
        print_success("All services stopped. Goodbye! 👋")
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)