import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import warnings
warnings.filterwarnings('ignore')

print("Testing ML imports...")

# Test 1: NumPy
try:
    import numpy as np
    print(f" NumPy {np.__version__}")
except Exception as e:
    print(f" NumPy failed: {e}")
    sys.exit(1)

# Test 2: scikit-learn
try:
    from sklearn.metrics.pairwise import cosine_similarity
    print(" scikit-learn")
except Exception as e:
    print(f" scikit-learn failed: {e}")
    sys.exit(1)

# Test 3: sentence-transformers (the problematic one)
try:
    from sentence_transformers import SentenceTransformer
    print(" sentence-transformers imported")
    print("   (Model will be downloaded on first use)")
except Exception as e:
    print(f" sentence-transformers failed: {e}")
    print("\nTrying to fix by installing torch without CUDA...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])
    print(" PyTorch CPU installed, retry importing...")
    try:
        from sentence_transformers import SentenceTransformer
        print(" sentence-transformers now working!")
    except Exception as e2:
        print(f" Still failed: {e2}")
        sys.exit(1)

# Test 4: ai_cv_analyzer
try:
    from ai_cv_analyzer import AdvancedCVExtractor
    print(" AdvancedCVExtractor imported")
    
    extractor = AdvancedCVExtractor()
    print(f" Extractor initialized")
    print(f"   - Has embedder: {extractor.embedder is not None}")
    print(f"   - Has NLP: {extractor.nlp is not None}")
    
    if extractor.embedder:
        print(" ML is READY! SentenceTransformer loaded")
    else:
        print(" ML not available, will use rule-based extraction")
        
except Exception as e:
    print(f" ai_cv_analyzer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print(" IMPORT TEST COMPLETE")
print("="*60)
