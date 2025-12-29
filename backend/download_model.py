"""
Download ML Model for Advanced ML Modules
This script pre-downloads the paraphrase-mpnet-base-v2 model to cache.
Run this once, and all subsequent tests will be instant.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print(" DOWNLOADING MODEL FOR ADVANCED ML MODULES")
print("=" * 70)
print("\nModel: paraphrase-mpnet-base-v2 (438MB)")
print("This is a ONE-TIME download. Once cached, all tests will be instant.\n")

try:
    from sentence_transformers import SentenceTransformer
    
    print("Starting download...")
    print("(This may take 2-5 minutes depending on your connection)\n")
    
    embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
    
    print("\n" + "=" * 70)
    print(" MODEL DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel: {embedder}")
    print(f"Device: {embedder.device}")
    print(f"Max sequence length: {embedder.max_seq_length}")
    
    print("\n You can now run:")
    print("   python test_advanced_ml_final.py")
    print("   python test_advanced_ml.py")
    print("   python test_ml_with_existing_parser.py")
    print("\nAll tests will run instantly! ")
    
except KeyboardInterrupt:
    print("\n  Download interrupted")
    print("Run this script again to resume download")
except Exception as e:
    print(f"\n Error: {e}")
    import traceback
    traceback.print_exc()
