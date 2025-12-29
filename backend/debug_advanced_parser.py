"""
Debug Advanced CV Parser
"""

print("Testing advanced_cv_parser.py import and initialization...")

try:
    print("\n1. Importing AdvancedCVParser...")
    from advanced_cv_parser import AdvancedCVParser
    print("    Import successful")
    
    print("\n2. Initializing parser...")
    parser = AdvancedCVParser()
    print("    Parser initialized")
    
    print("\n3. Testing parse_cv with simple text...")
    test_cv = "Senior engineer with Python, AWS. Led team of 5."
    
    try:
        result = parser.parse_cv(test_cv)
        print(f"    Parse successful!")
        print(f"   Skills: {result.skills}")
        print(f"   Seniority: {result.seniority_level}")
        print(f"   Industries: {result.industries}")
    except Exception as e:
        print(f"    Parse failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f" Failed: {e}")
    import traceback
    traceback.print_exc()
