# AI-Powered Interview Feature - Integration Documentation

## Overview

This document describes the AI-powered interview feature that has been integrated into SkillSync. This feature allows users to practice interviews based on their CV and a target job description, and receive AI-generated feedback on their performance.

## Architecture

### Backend Components

#### 1. Interview Agent (`backend/agents/interview_agent.py`)
The core logic for managing interview sessions:
- **Question Generation**: Generates tailored interview questions based on CV and job description
- **Session Management**: Tracks interview progress and stores Q&A pairs
- **Performance Analysis**: Analyzes responses and provides feedback (currently mocked)

**Key Methods:**
- `start_interview(cv_text, job_description, num_questions)` - Initiates a new interview session
- `submit_answer(interview_id, question_id, answer_text)` - Records an answer
- `get_report(interview_id)` - Generates complete performance report

#### 2. Data Models (`backend/models/interview_models.py`)
Pydantic models for type-safe API contracts:
- `StartInterviewRequest` / `StartInterviewResponse`
- `SubmitAnswerRequest` / `SubmitAnswerResponse`
- `InterviewReportResponse`
- `InterviewAnalysis`
- `InterviewTranscriptItem`

#### 3. API Router (`backend/routers/interview_router.py`)
FastAPI router exposing three endpoints:

```python
POST   /api/v1/interviews/start
POST   /api/v1/interviews/{interview_id}/submit_answer
GET    /api/v1/interviews/{interview_id}/report
```

#### 4. Main Application (`backend/main.py`)
Simplified FastAPI application that includes:
- CORS middleware configuration
- Interview router integration
- Health check endpoints

### Frontend Components

#### 1. Interview Service (`frontend/src/services/interviewService.ts`)
TypeScript service for backend communication with strongly-typed functions:
- `startInterview(data)` - Start new interview
- `submitAnswer(interviewId, data)` - Submit answer
- `getReport(interviewId)` - Fetch report

#### 2. Interview Page (`frontend/src/pages/Interview/InterviewPage.tsx`)
Interactive interview interface with:
- **Pre-Interview Screen**: Input CV and job description
- **Question Interface**: Dynamic Q&A with progress tracking
- **Mock Recording**: Voice recording placeholder (Web Speech API integration point)
- **Animations**: Smooth transitions between questions using Framer Motion

#### 3. Interview Report Page (`frontend/src/pages/Interview/InterviewReportPage.tsx`)
Comprehensive performance report displaying:
- Overall performance score with color-coded visualization
- Summary of performance
- Strengths and weaknesses breakdown
- Improvement recommendations
- Complete interview transcript

#### 4. Routing Integration
- Added routes in `App.tsx`:
  - `/interview` - Main interview page
  - `/interview-report/:interviewId` - Report page
- Added "Interview Practice" link to sidebar navigation

## API Endpoints

### 1. Start Interview
```http
POST /api/v1/interviews/start
Content-Type: application/json

{
  "cv_text": "Your CV content...",
  "job_description": "Job description...",
  "num_questions": 5
}
```

**Response:**
```json
{
  "interview_id": "uuid",
  "questions": [
    {
      "question_id": 1,
      "question_text": "Question text",
      "category": "technical"
    }
  ],
  "message": "Interview session started successfully"
}
```

### 2. Submit Answer
```http
POST /api/v1/interviews/{interview_id}/submit_answer
Content-Type: application/json

{
  "question_id": 1,
  "answer_text": "Your answer..."
}
```

**Response:**
```json
{
  "message": "Answer submitted successfully",
  "next_question": {
    "question_id": 2,
    "question_text": "Next question...",
    "category": "behavioral"
  },
  "is_complete": false
}
```

### 3. Get Report
```http
GET /api/v1/interviews/{interview_id}/report
```

**Response:**
```json
{
  "interview_id": "uuid",
  "cv_text": "CV content",
  "job_description": "Job description",
  "transcript": [...],
  "analysis": {
    "overall_score": 85.5,
    "summary": "Summary text...",
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"]
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Testing

### Backend Testing
Run the provided test script:
```bash
chmod +x test_interview_endpoints.sh
./test_interview_endpoints.sh
```

Or test manually:
```bash
# Start the backend server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Test endpoints using curl (see test script for examples)
```

### Frontend Testing
1. Ensure backend is running on port 8000
2. Start frontend development server:
```bash
cd frontend
npm run dev
```
3. Navigate to http://localhost:5173/interview
4. Complete an interview flow:
   - Enter CV and job description
   - Answer questions
   - View report

## Future Enhancements

### AI Integration
The current implementation uses mock AI logic. To integrate with a real AI service:

1. **Question Generation**: Replace mock questions in `interview_agent.py` with calls to:
   - Google AI / Gemini
   - OpenAI GPT-4
   - Anthropic Claude
   - Custom fine-tuned models

2. **Analysis & Scoring**: Implement real AI-based analysis:
   - NLP-based response evaluation
   - STAR method detection
   - Keyword matching with job requirements
   - Sentiment analysis
   - Answer completeness scoring

3. **Voice Recording**: Implement Web Speech API in frontend:
   - Speech-to-text transcription
   - Automatic answer population
   - Pronunciation feedback
   - Filler word detection

### Advanced Features
- **Video Recording**: Add webcam support for body language analysis
- **Multi-language Support**: Support interviews in different languages
- **Industry-specific Templates**: Pre-built question sets for different industries
- **Progress Tracking**: Track improvement across multiple interview sessions
- **Share Reports**: Export and share interview reports
- **Collaborative Mode**: Allow coaches/mentors to review and comment

## Security Considerations

- All endpoints should be protected with authentication in production
- Implement rate limiting to prevent abuse
- Sanitize user inputs (CV and job descriptions)
- Store interview data securely with encryption
- Implement data retention policies
- Add GDPR compliance features (data export, deletion)

## Deployment

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
cd frontend
npm install
npm run build
# Deploy dist/ folder to static hosting (Netlify, Vercel, etc.)
```

### Environment Variables
Backend `.env`:
```
DEBUG=false
LOG_LEVEL=info
# Add AI service API keys when implementing real AI
OPENAI_API_KEY=your_key_here
GOOGLE_AI_API_KEY=your_key_here
```

## Contributing

When contributing to the interview feature:
1. Follow existing code structure and patterns
2. Add tests for new functionality
3. Update this documentation
4. Ensure security best practices
5. Test the full user flow before submitting PR

## License

This feature is part of the SkillSync project and follows the same MIT license.
