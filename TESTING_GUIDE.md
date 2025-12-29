# SkillSync Backend - Complete Testing Guide

## Quick Test Commands

### 1. Backend Unit Tests
```cmd
cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\backend
pytest tests/ -v
```

### 2. Backend Integration Tests  
```cmd
python test_backend_only.py
python test_recommendations.py
python test_job_apis.py
```

### 3. Specific CV Flow Tests
```cmd
pytest tests/test_cv_flows.py -v
```

### 4. Frontend TypeScript Check
```cmd
cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced\frontend
npm run type-check
npm run lint
```

### 5. Full E2E Manual Test
1. Start backend:
   ```cmd
   cd backend
   python main_simple_for_frontend.py
   ```

2. Start frontend:
   ```cmd
   cd frontend
   npm run dev
   ```

3. Open browser: `http://localhost:5173`

4. Test flows:
   - Upload CV → see analysis
   - Click "View Recommendations"
   - Check Dashboard metrics
   - Generate CV/Portfolio
   - Verify no static data

## Expected Results

### ✅ All tests should pass with:
- No 404 errors on `/api/v1/cv-analyses`
- No `Cannot read properties of undefined` errors
- Dashboard shows real data (not 8/42/76)
- Recommendations work with analysisId
- CV generation downloads actual file

### ❌ Known Issues (Acceptable):
- ML backend warnings (TensorFlow/PyTorch) - falls back to rules
- MessageFactory protobuf warning - non-fatal
- Some job APIs may timeout - graceful degradation

## Coverage Goals
- Backend endpoints: 80%+
- Critical flows: 100%
- Frontend components: 70%+
