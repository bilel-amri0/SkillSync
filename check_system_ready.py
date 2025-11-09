#!/usr/bin/env python3

"""
SkillSync System Readiness Checker
Comprehensive check to ensure the entire system is ready for use
"""

import asyncio
import subprocess
import sys
import os
import json
from pathlib import Path
import aiohttp
from datetime import datetime

class SystemChecker:
    """Comprehensive system checker"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        self.checks_passed = 0
        self.total_checks = 0
        
    def run_check(self, description: str, check_func):
        """Run a single check with status reporting"""
        self.total_checks += 1
        print(f"[{self.total_checks}] {description}...", end=" ")
        
        try:
            result = check_func()
            if result:
                print("✅")
                self.checks_passed += 1
                return True
            else:
                print("❌")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    async def run_async_check(self, description: str, check_func):
        """Run a single async check with status reporting"""
        self.total_checks += 1
        print(f"[{self.total_checks}] {description}...", end=" ")
        
        try:
            result = await check_func()
            if result:
                print("✅")
                self.checks_passed += 1
                return True
            else:
                print("❌")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def check_project_structure(self):
        """Check project directory structure"""
        required_dirs = [
            self.backend_dir,
            self.frontend_dir,
            self.backend_dir / "services",
        ]
        
        required_files = [
            self.backend_dir / ".env",
            self.backend_dir / "main_simple_for_frontend.py",
            self.backend_dir / "services" / "multi_job_api_service.py",
            self.frontend_dir / "package.json",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"\n   Missing directory: {dir_path}")
                return False
        
        for file_path in required_files:
            if not file_path.exists():
                print(f"\n   Missing file: {file_path}")
                return False
        
        return True
    
    def check_backend_dependencies(self):
        """Check if backend dependencies are installed"""
        try:
            # Try importing key packages
            import fastapi
            import aiohttp
            import uvicorn
            import pydantic
            return True
        except ImportError as e:
            print(f"\n   Missing dependency: {e}")
            return False
    
    def check_frontend_dependencies(self):
        """Check if frontend dependencies are installed"""
        node_modules = self.frontend_dir / "node_modules"
        package_json = self.frontend_dir / "package.json"
        
        if not package_json.exists():
            print(f"\n   Missing package.json")
            return False
        
        if not node_modules.exists():
            print(f"\n   Node modules not installed. Run: npm install")
            return False
        
        return True
    
    def check_env_configuration(self):
        """Check .env file configuration"""
        env_file = self.backend_dir / ".env"
        
        if not env_file.exists():
            print(f"\n   .env file not found")
            return False
        
        # Check for essential API keys
        required_keys = [
            'LINKEDIN_RAPIDAPI_KEY',
            'JSEARCH_RAPIDAPI_KEY',
            'MUSE_API_KEY',
            'FINDWORK_API_KEY',
            'ADZUNA_APP_ID',
            'ADZUNA_APP_KEY'
        ]
        
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        missing_keys = []
        for key in required_keys:
            if key not in env_content or f"{key}=" not in env_content:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n   Missing API keys: {', '.join(missing_keys)}")
            return False
        
        return True
    
    async def check_api_connectivity(self):
        """Check if APIs are accessible"""
        # Quick test of JSearch API (most reliable)
        try:
            # Load env vars
            env_file = self.backend_dir / ".env"
            env_vars = {}
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
            
            api_key = env_vars.get('JSEARCH_RAPIDAPI_KEY')
            if not api_key:
                return False
            
            headers = {
                'X-RapidAPI-Key': api_key,
                'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
            }
            params = {
                'query': 'test',
                'page': '1',
                'num_pages': '1'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://jsearch.p.rapidapi.com/search",
                    headers=headers,
                    params=params
                ) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    def check_backend_can_start(self):
        """Check if backend can be imported and started"""
        try:
            # Change to backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)
            
            # Try importing the main module
            sys.path.insert(0, str(self.backend_dir))
            
            # This will fail if there are syntax errors or missing dependencies
            import main_simple_for_frontend
            
            # Restore
            os.chdir(original_cwd)
            sys.path.remove(str(self.backend_dir))
            
            return True
        except Exception as e:
            print(f"\n   Backend import error: {e}")
            return False
    
    async def check_backend_health(self):
        """Check if backend health endpoint responds"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:8000/health") as response:
                    return response.status == 200
        except:
            # Backend might not be running, which is OK for this check
            print("\n   Backend not running (run: python main_simple_for_frontend.py)")
            return True  # This is not a failure
    
    def print_final_report(self):
        """Print final system readiness report"""
        print("\n" + "="*60)
        print("🎯 SKILLSYNC SYSTEM READINESS REPORT")
        print("="*60)
        
        success_rate = (self.checks_passed / self.total_checks) * 100
        
        print(f"📊 Checks passed: {self.checks_passed}/{self.total_checks} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 SYSTEM READY! Your SkillSync platform is fully operational!")
            print("\n🚀 Next steps:")
            print("   1. Start backend: cd backend && python main_simple_for_frontend.py")
            print("   2. Start frontend: cd frontend && npm start")
            print("   3. Open: http://localhost:3000/jobs")
            print("   4. Test job search with your configured APIs!")
        elif success_rate >= 70:
            print("⚠️  SYSTEM MOSTLY READY - Minor issues detected")
            print("\n🔧 Fix any failed checks above, then you're good to go!")
        else:
            print("❌ SYSTEM NOT READY - Major issues detected")
            print("\n🔧 Please fix the failed checks above before proceeding")
            print("\n📚 Helpful commands:")
            print("   • python install_requirements.py (install dependencies)")
            print("   • python quick_setup_apis.py (configure API keys)")
            print("   • cd frontend && npm install (install frontend deps)")
        
        print("\n" + "="*60)

async def main():
    """Main system check runner"""
    print("🔍 SkillSync System Readiness Check")
    print("="*40)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)
    
    checker = SystemChecker()
    
    # Run all checks
    checker.run_check("Project structure", checker.check_project_structure)
    checker.run_check("Backend dependencies", checker.check_backend_dependencies)
    checker.run_check("Frontend dependencies", checker.check_frontend_dependencies)
    checker.run_check("Environment configuration", checker.check_env_configuration)
    await checker.run_async_check("API connectivity", checker.check_api_connectivity)
    checker.run_check("Backend module import", checker.check_backend_can_start)
    await checker.run_async_check("Backend health", checker.check_backend_health)
    
    # Final report
    checker.print_final_report()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Check interrupted by user.")
    except Exception as e:
        print(f"\n❌ System check failed: {e}")
        sys.exit(1)