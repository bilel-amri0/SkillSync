"""
Installation and setup script for SkillSync backend
Handles environment setup and dependency installation
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(f" {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("""
    
             SkillSync Backend Setup & Installation          
    
    """)
    
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    print("\n Working directory:", backend_dir)
    print(" Python version:", sys.version)
    
    # Step 1: Create virtual environment (optional but recommended)
    create_venv = input("\n Create virtual environment? (y/n) [y]: ").lower()
    if create_venv != 'n':
        if not run_command("python -m venv venv", "Creating virtual environment"):
            print("\n  Virtual environment creation failed. Continuing with system Python...")
        else:
            print("\n Activate virtual environment:")
            if sys.platform == "win32":
                print("   Windows: venv\\Scripts\\activate")
            else:
                print("   Unix/Mac: source venv/bin/activate")
            
            activate = input("\n Continue with installation? (y/n) [y]: ").lower()
            if activate == 'n':
                print("\n Exiting. Please activate venv and run this script again.")
                return
    
    # Step 2: Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 3: Install core dependencies
    requirements_file = "requirements-fixed.txt" if os.path.exists("requirements-fixed.txt") else "requirements.txt"
    
    if not run_command(
        f"{sys.executable} -m pip install -r {requirements_file}",
        f"Installing core dependencies from {requirements_file}"
    ):
        print("\n Failed to install core dependencies. Please check the error above.")
        return
    
    # Step 4: Ask about ML dependencies
    install_ml = input("\n Install ML dependencies (PyTorch, SBERT, SpaCy)? This is optional. (y/n) [n]: ").lower()
    if install_ml == 'y':
        if os.path.exists("requirements-ml.txt"):
            if run_command(
                f"{sys.executable} -m pip install -r requirements-ml.txt",
                "Installing ML dependencies"
            ):
                # Download SpaCy model
                run_command(
                    f"{sys.executable} -m spacy download en_core_web_sm",
                    "Downloading SpaCy English model"
                )
        else:
            print("\n  requirements-ml.txt not found. Skipping ML installation.")
    
    # Step 5: Create .env if it doesn't exist
    if not os.path.exists(".env"):
        print("\n Creating .env file...")
        with open(".env", "w") as f:
            f.write("""# SkillSync Backend Configuration
# Database
DATABASE_URL=sqlite:///./skillsync.db

# API Keys (optional - for job search APIs)
JSEARCH_RAPIDAPI_KEY=your_jsearch_key_here
ADZUNA_APP_ID=your_adzuna_id_here
ADZUNA_APP_KEY=your_adzuna_key_here

# CORS (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Server
HOST=127.0.0.1
PORT=8001
""")
        print(" .env file created. Please edit it with your API keys.")
    else:
        print("\n .env file already exists.")
    
    # Step 6: Run tests
    run_tests = input("\n Run tests to verify installation? (y/n) [y]: ").lower()
    if run_tests != 'n':
        if os.path.exists("tests"):
            run_command(f"{sys.executable} -m pytest tests/ -v", "Running tests")
        else:
            print("\n  No tests folder found. Skipping tests.")
    
    print("""
    
                       Setup Complete!                    
    
    
     Next steps:
    
    1. Edit .env file with your API keys (optional)
    2. Start the backend server:
       
       python main_simple_for_frontend.py
       
    3. Access the API documentation:
       http://127.0.0.1:8001/api/docs
    
    4. Run tests:
       pytest tests/ -v
    
     Tips:
    - Use requirements-fixed.txt for stable dependencies
    - Install ML deps only if you need semantic matching
    - Check logs for any startup warnings
    
     Documentation: See README.md for more details
    """)


if __name__ == "__main__":
    main()
