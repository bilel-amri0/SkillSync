import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Testing ML Integration")
print("=" * 60)

# Test 1: Import packages
try:
    import numpy as np
    print(" NumPy imported:", np.__version__)
except Exception as e:
    print(" NumPy failed:", e)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print(" scikit-learn imported")
except Exception as e:
    print(" scikit-learn failed:", e)

try:
    from sentence_transformers import SentenceTransformer
    print(" sentence-transformers imported")
except Exception as e:
    print(" sentence-transformers failed:", e)
    sys.exit(1)

# Test 2: Load model
print("\n" + "=" * 60)
print("Loading SentenceTransformer model...")
print("=" * 60)

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(" Model loaded successfully")
    print(f"   Model type: {type(model)}")
    print(f"   Max sequence length: {model.max_seq_length}")
except Exception as e:
    print(" Model loading failed:", e)
    sys.exit(1)

# Test 3: Encode text
print("\n" + "=" * 60)
print("Testing encoding...")
print("=" * 60)

try:
    test_text = "Python Machine Learning Engineer"
    embedding = model.encode(test_text)
    print(f" Encoding successful")
    print(f"   Input: {test_text}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding type: {type(embedding)}")
except Exception as e:
    print(" Encoding failed:", e)
    sys.exit(1)

# Test 4: Test ai_cv_analyzer
print("\n" + "=" * 60)
print("Testing AI CV Analyzer...")
print("=" * 60)

try:
    from ai_cv_analyzer import AdvancedCVExtractor
    print(" AdvancedCVExtractor imported")
    
    extractor = AdvancedCVExtractor()
    print(" Extractor initialized")
    print(f"   Has embedder: {extractor.embedder is not None}")
    print(f"   Has NLP: {extractor.nlp is not None}")
    
    # Test parsing
    test_cv = """
    John Doe
    john.doe@email.com
    +1-555-1234
    
    Senior Python Developer
    
    Skills:
    - Python
    - Machine Learning
    - TensorFlow
    - React
    - Docker
    
    Experience:
    Software Engineer at Tech Corp (2020-2023)
    - Developed ML models
    - Built web applications
    
    Education:
    Bachelor of Computer Science, MIT, 2020
    """
    
    cv_data = extractor.parse_cv_advanced(test_cv)
    print(" CV parsed successfully")
    print(f"   Name: {cv_data.name}")
    print(f"   Email: {cv_data.email}")
    print(f"   Skills: {cv_data.skills}")
    print(f"   Confidence scores: {cv_data.confidence_scores}")
    
except Exception as e:
    print(" AI CV Analyzer failed:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print(" ALL TESTS PASSED - ML IS WORKING!")
print("=" * 60)
