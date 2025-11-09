#!/usr/bin/env python3
"""Test script for SkillSync database setup and operations"""

import sys
import json
from datetime import datetime
from database import (
    init_db, get_db_session, test_connection,
    CVAnalysisService, UserService, SkillService, RecommendationService,
    get_database_stats
)

def test_database_setup():
    """Test basic database setup"""
    print("🔧 Testing database setup...")
    
    # Test connection
    if not test_connection():
        print("❌ Database connection failed")
        return False
    print("✅ Database connection successful")
    
    # Initialize tables
    if not init_db():
        print("❌ Database initialization failed")
        return False
    print("✅ Database tables created")
    
    return True

def test_basic_operations():
    """Test basic CRUD operations"""
    print("\n📝 Testing basic CRUD operations...")
    
    db = get_db_session()
    
    try:
        # Test 1: Create anonymous user
        user = UserService.get_or_create_anonymous_user(db)
        print(f"✅ Created user: {user.id}")
        
        # Test 2: Create CV analysis with mock data
        mock_analysis_data = {
            "immediate_actions": [
                "Update LinkedIn profile with recent projects",
                "Add quantifiable achievements to resume"
            ],
            "skill_development": [
                "Learn advanced Python frameworks like FastAPI",
                "Improve data visualization skills"
            ],
            "project_ideas": [
                "Build a portfolio website",
                "Create an open-source project"
            ],
            "learning_resources": [
                "Python documentation and tutorials",
                "Online courses for specific skills"
            ],
            "career_roadmap": [
                "Apply for junior developer positions",
                "Build a strong portfolio"
            ]
        }
        
        analysis = CVAnalysisService.create_analysis(
            db=db,
            filename="test_cv.pdf",
            original_text="Sample CV content for testing...",
            analysis_data=mock_analysis_data,
            user_id=user.id
        )
        print(f"✅ Created analysis: {analysis.id}")
        
        # Test 3: Create skills from analysis
        mock_skills = [
            {"name": "Python", "category": "technical", "level": "intermediate", "confidence": 0.9},
            {"name": "JavaScript", "category": "technical", "level": "beginner", "confidence": 0.7},
            {"name": "Communication", "category": "soft", "level": "advanced", "confidence": 0.8}
        ]
        
        skills = SkillService.create_skills_from_analysis(
            db=db,
            analysis_id=analysis.id,
            skills_data=mock_skills
        )
        print(f"✅ Created {len(skills)} skills")
        
        # Test 4: Create recommendations
        recommendations = RecommendationService.create_recommendations_from_analysis(
            db=db,
            analysis_id=analysis.id,
            recommendations_data=mock_analysis_data
        )
        print(f"✅ Created {len(recommendations)} recommendations")
        
        # Test 5: Retrieve data
        retrieved_analysis = CVAnalysisService.get_analysis(db, analysis.id)
        if retrieved_analysis:
            print(f"✅ Retrieved analysis: {retrieved_analysis.filename}")
        
        retrieved_recommendations = RecommendationService.get_analysis_recommendations(db, analysis.id)
        print(f"✅ Retrieved recommendations: {len(retrieved_recommendations)} categories")
        
        # Test 6: Database statistics
        stats = get_database_stats(db)
        print(f"✅ Database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ CRUD operations failed: {e}")
        return False
    finally:
        db.close()

def test_api_integration():
    """Test integration with existing API format"""
    print("\n🔌 Testing API integration...")
    
    db = get_db_session()
    
    try:
        # Simulate API workflow
        
        # 1. Get latest analysis (like API endpoint would)
        latest_analysis = CVAnalysisService.get_latest_analysis(db)
        
        if latest_analysis:
            print(f"✅ Latest analysis found: {latest_analysis.id}")
            
            # 2. Get recommendations in API format
            recommendations = RecommendationService.get_analysis_recommendations(db, latest_analysis.id)
            
            # 3. Verify format matches what React expects
            expected_categories = ['immediate_actions', 'skill_development', 'project_ideas', 
                                 'learning_resources', 'career_roadmap']
            
            missing_categories = [cat for cat in expected_categories if cat not in recommendations]
            if missing_categories:
                print(f"⚠️  Missing categories: {missing_categories}")
            else:
                print("✅ All expected categories present")
            
            # 4. Show sample output
            print("\n📊 Sample API Response Format:")
            for category, items in recommendations.items():
                print(f"  {category}: {len(items)} items")
                if items:
                    print(f"    - {items[0]['title'][:50]}...")
            
            return True
        else:
            print("❌ No analysis found for API integration test")
            return False
            
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False
    finally:
        db.close()

def show_next_steps():
    """Show next steps for integration"""
    print("\n🚀 Next Steps for API Integration:")
    print("\n1. Update your FastAPI endpoints to use database:")
    print("   - Add 'from database import CVAnalysisService, RecommendationService'")
    print("   - Replace mock responses with database queries")
    print("\n2. Example endpoint modification:")
    print("""
@app.post("/api/v1/upload-cv")
async def upload_cv(file: UploadFile):
    # Your existing processing...
    
    # Save to database
    db = get_db_session()
    analysis = CVAnalysisService.create_analysis(
        db=db,
        filename=file.filename,
        original_text=cv_text,
        analysis_data=result
    )
    db.close()
    
    return {"analysis_id": analysis.id, **result}
    """)
    
    print("\n3. Add database dependency to main FastAPI app")
    print("\n4. Test with React frontend")
    
    print("\n📋 Files created:")
    print("   ✅ models.py - Database schema (Option B ready)")
    print("   ✅ database.py - CRUD operations")
    print("   ✅ test_db.py - This test script")
    print("   ✅ skillsync.db - SQLite database (after running this script)")

def main():
    """Main test function"""
    print("🎯 SkillSync Database Setup & Test")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_database_setup()
    if success:
        success &= test_basic_operations()
    if success:
        success &= test_api_integration()
    
    # Show results
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nDatabase is ready for integration.")
        show_next_steps()
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
