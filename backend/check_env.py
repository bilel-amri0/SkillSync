#!/usr/bin/env python3
"""
Quick environment check script
"""
import os
from pathlib import Path
from dotenv import load_dotenv
def main():
    print("🔍 SkillSync Environment Configuration Check")
    print("=" * 45)
    
    # Load .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Found .env file: {env_file}")
    else:
        print("❌ .env file not found!")
        return
    
    # Check API keys
    api_keys = {
        'LinkedIn RapidAPI': 'LINKEDIN_RAPIDAPI_KEY',
        'JSearch RapidAPI': 'JSEARCH_RAPIDAPI_KEY', 
        'The Muse': 'MUSE_API_KEY',
        'FindWork': 'FINDWORK_API_KEY',
        'Adzuna App ID': 'ADZUNA_APP_ID',
        'Adzuna App Key': 'ADZUNA_APP_KEY'
    }
    
    print("\n📋 API Keys Status:")
    configured_count = 0
    
    for name, env_var in api_keys.items():
        value = os.getenv(env_var)
        if value:
            print(f"   ✅ {name:<20} : {'*' * 10}...{value[-4:]}")
            configured_count += 1
        else:
            print(f"   ❌ {name:<20} : NOT FOUND")
    
    print(f"\n📊 Summary: {configured_count}/{len(api_keys)} API keys configured")
    
    if configured_count == len(api_keys):
        print("🎉 All API keys are configured!")
        print("\n🚀 Next step: python test_all_apis.py")
    else:
        print(f"⚠️  {len(api_keys) - configured_count} API keys missing")
        print("\n🔧 Next step: python quick_setup_apis.py")
    
    print("\n" + "=" * 45)
if __name__ == "__main__":
    main()