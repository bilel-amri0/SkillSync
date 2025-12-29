# SkillSync API - Quick Fix Guide

## Issue Encountered

**Error 1: Import Error**
```
NameError: name 'Image' is not defined
```

**Error 2: Dependency Conflict**
```
ERROR: Cannot install langchain==0.0.350 and numpy==2.2.1 because these package versions have conflicting dependencies.
```

## Solutions Applied

### Fix 1: cv_processor.py Import Guards

Updated import statements to handle missing dependencies gracefully:

```python
# Before (caused NameError)
try:
    from PIL import Image
    ADVANCED_PDF_AVAILABLE = True
except ImportError:
    ADVANCED_PDF_AVAILABLE = False

# After (fixed)
try:
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    Image = None  # Fallback for type hints
```

Also changed type hint from `Image.Image` to `Any` to avoid runtime errors.

### Fix 2: Dependency Versions

**Updated requirements.txt:**
- Changed `numpy==2.2.1` → `numpy==1.26.2` (compatible with langchain)
- Changed `pandas==2.2.3` → `pandas==2.1.4` (compatible with numpy 1.26)

**Created requirements_minimal.txt:**
Only installs essentials for Phase 1.5:
- FastAPI + uvicorn
- PyPDF2 (PDF parsing)
- python-docx (DOCX parsing)
- Jinja2 (portfolio templates)
- python-multipart (file uploads)

## Quick Start (FIXED)

### Option 1: Minimal Install (Recommended for testing)
```cmd
cd backend
pip install -r requirements_minimal.txt
python main.py
```

### Option 2: Full Install (All features)
```cmd
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python main.py
```

### Option 3: Use Fixed Batch Script
```cmd
start_api_fixed.bat
```

## What Works with Minimal Install

✅ **API Server**: FastAPI runs perfectly
✅ **PDF Parsing**: PyPDF2 extracts text from text-based PDFs
✅ **DOCX Parsing**: python-docx extracts from Word documents
✅ **Portfolio Generation**: Jinja2 templates work fully
✅ **Health Checks**: All status endpoints work

## What Requires Full Install

❌ **OCR (Scanned PDFs)**: Needs pdf2image, pytesseract, Pillow, opencv-python
❌ **NLP Skill Extraction**: Needs spacy + en_core_web_lg model
❌ **Semantic Matching**: Needs sentence-transformers, scikit-learn

## Testing

After starting the server, test with:

```cmd
# Terminal 1: Start server
python backend/main.py

# Terminal 2: Run tests
python test_api_phase_1_5.py
```

Or visit: http://localhost:8000/api/docs

## Files Modified

1. ✅ `backend/services/cv_processor.py` - Fixed Image import
2. ✅ `backend/requirements.txt` - Fixed numpy version
3. ✅ `backend/requirements_minimal.txt` - Created minimal deps
4. ✅ `start_api_fixed.bat` - Fixed startup script

## Next Steps

1. Try starting with `start_api_fixed.bat`
2. Test API at http://localhost:8000/api/docs
3. If everything works, optionally install full dependencies for advanced features

## Error Prevention

The code now handles missing dependencies gracefully:
- If PyPDF2 missing → Falls back to text file parsing only
- If spaCy missing → Uses pattern-based skill extraction
- If Jinja2 missing → Returns error with clear message
- No more NameError crashes!
