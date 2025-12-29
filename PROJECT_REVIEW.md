# SkillSync Project Review

## 1. Project Structure & Organization
**Rating: ⭐⭐⭐ (3/5)**

The project has a clear high-level separation between `frontend` and `backend`, which is excellent. However, the root directory is heavily cluttered with over 40 markdown files, making it difficult to navigate.

### ✅ Strengths
- Clear separation of concerns (`frontend/` vs `backend/`).
- Standard directory structure for React and Python projects.

### ⚠️ Issues
- **Root Clutter**: The root directory contains an overwhelming number of documentation files, logs, and reports (e.g., `*_SUMMARY.md`, `*_GUIDE.md`).
- **Duplicate Entry Points**: The backend has multiple entry points (`main.py`, `main_enhanced.py`, `main_ml.py`, `main_simple_for_frontend.py`), leading to confusion about which one is the source of truth.

## 2. Tech Stack
**Rating: ⭐⭐⭐⭐ (4/5)**

The technology choices are modern and appropriate for the task, though the backend dependencies need pruning.

### Frontend
- **Framework**: React 19 + Vite 7 (Cutting edge).
- **Language**: TypeScript (Excellent for maintainability).
- **Styling**: Tailwind CSS (Modern standard).
- **State/Data**: React Query (Best practice for API data).

### Backend
- **Framework**: FastAPI (Excellent choice for performance and DX).
- **Language**: Python 3.11+.
- **Dependencies**: The `requirements.txt` file is bloated (280+ lines), including heavy libraries like TensorFlow, PyTorch, and Transformers that may not be fully utilized or necessary for the core functionality.

## 3. Code Quality
**Rating: ⭐⭐⭐ (3/5)**

### Frontend
The frontend follows modern practices and looks clean.

### Backend
The backend is in a transitional state.
- **`main.py`**: Follows a modular architecture, importing routers and middleware. This is the correct *architectural* approach but appears to lack some features present in the "simple" version.
- **`main_simple_for_frontend.py`**: This is the **actual working entry point** used by the current workflow. It is a massive monolithic file (3500+ lines) that contains:
    - Complex ML model loading with fallbacks (Lite/Full/Hybrid).
    - Hardcoded business logic for career roadmaps and recommendations.
    - Custom endpoints not found in the modular structure.
    - **Risk**: This file is hard to maintain and test due to its size and complexity.

## 4. Recommendations

### 🚨 Critical: Entry Point Confusion
You are currently using `main_simple_for_frontend.py`, which is the correct choice for functionality right now because it contains features missing from the modular `main.py`. **Do not switch to `main.py` yet without a migration plan.**

**Note on Ports**:
- **Backend**: Runs on port **8001** (configured in `main_simple_for_frontend.py`).
- **Frontend**: Configured to talk to `http://localhost:8001` in `src/api.ts`.
- **Standard**: The modular `main.py` defaults to port **8000**. If you switch, you must update the frontend configuration.

### ✅ Resolved Issues
- **Frontend ReferenceError**: Fixed a crash in `App.tsx` where `handleJobSearch` was being called before initialization. The frontend is now fully functional.

### 🧹 Cleanup & Organization
1.  **Consolidate Documentation**: Move all `*.md` files from the root into a `docs/` directory. Keep only `README.md`.
2.  **Prune Dependencies**: Adopt `requirements_minimal.txt` as the base.

### 🛠️ Refactoring Plan (Long Term)
The goal should be to migrate the logic from `main_simple_for_frontend.py` into the modular structure of `main.py`.
1.  **Extract Services**: Move the ML logic, Career Roadmap generation, and Experience Translator from `main_simple_for_frontend.py` into dedicated service classes in `backend/services/`.
2.  **Update Routers**: Update the routers in `backend/routers/` to use these new services.
3.  **Switch Entry Point**: Once `main.py` has feature parity, switch to it and archive `main_simple_for_frontend.py`.

### 🚀 Next Steps
1.  Create a `docs` folder and move files to clean up the root.
2.  Create a `start_project.bat` script that formalizes your current workflow (running `main_simple_for_frontend.py` and `npm run dev`).
