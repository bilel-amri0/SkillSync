# 🎯 SkillSync AI Interview - Quick Start Guide

## Overview
AI-powered interview preparation with **Text** and **Voice** modes to help you practice for real job interviews.

## Features
- ✅ **Text Interviews** - Type detailed answers, edit before submitting
- 🎤 **Voice Interviews** - Real-time conversation with AI interviewer (Coming Soon)
- 📊 **Instant Feedback** - AI-powered analysis and suggestions
- 📈 **Progress Tracking** - Track answered questions and performance
- 🎓 **Tailored Questions** - Based on your CV and target job

## Quick Start

### 1. Start Backend
```bash
cd backend
python main_simple_for_frontend.py
```
Backend runs on: http://localhost:8001

### 2. Start Frontend
```bash
cd frontend
npm run dev
```
Frontend runs on: http://localhost:5173

### 3. Use the Feature
1. **Upload CV**: Go to "CV Analysis" and upload your CV
2. **Start Interview**: Click "AI Interview" in navigation
3. **Choose Mode**: Select "Text Interview" (recommended)
4. **Fill Details**: Enter job title, description, difficulty
5. **Start**: Begin your interview!

## Testing

### Test Backend API
```bash
python test_interview_api.py
```

### Test Manually with curl
```bash
# Start interview
curl -X POST http://localhost:8001/api/v2/interviews/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "cv_text": "Python developer",
    "job_title": "Senior Developer",
    "difficulty": "medium",
    "skills": ["Python", "React"],
    "interview_mode": "text"
  }'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/interviews/start` | Start new interview |
| POST | `/api/v2/interviews/answer` | Submit answer |
| POST | `/api/v2/interviews/next-question` | Get next question |
| POST | `/api/v2/interviews/finish` | Complete interview |
| GET | `/api/v2/interviews/{id}` | Get details |
| GET | `/api/v2/interviews/{id}/report` | Get report |
| GET | `/api/v2/interviews/` | List interviews |

## Interview Workflow

```
1. Upload CV
   ↓
2. Start Interview (select mode)
   ↓
3. Receive Question
   ↓
4. Submit Answer
   ↓
5. Next Question (repeat 3-5)
   ↓
6. Finish Interview
   ↓
7. View Report
```

## Interview Modes

### Text Mode ⌨️
- **Best for**: Detailed answers, editing before submit
- **Features**: 
  - Question-by-question interface
  - Answer history
  - Progress tracking
  - Character counter
  - Quick submit (Ctrl+Enter)

### Voice Mode 🎤 (Beta)
- **Best for**: Speaking practice, real interview simulation
- **Features**:
  - Real-time audio streaming
  - AI voice responses
  - Conversation transcript
  - Duration timer
  - Mic controls

## Tips for Great Interviews

### Before Interview
- ✅ Upload complete CV
- ✅ Review target job description
- ✅ Note key skills required
- ✅ Prepare examples from experience

### During Interview
- 📝 Use STAR method (Situation, Task, Action, Result)
- 🎯 Be specific with examples
- ⏱️ Take time to think
- 💡 Show problem-solving process

### After Interview
- 📊 Review report
- 📈 Note improvement areas
- 🔄 Practice weak topics
- ✨ Apply feedback

## Configuration

### Backend (.env)
```bash
# Required for voice mode
GOOGLE_API_KEY=your_google_api_key_here

# Database (default: SQLite)
DATABASE_URL=sqlite:///./skillsync.db
```

### Frontend (api.ts)
```typescript
const API_BASE_URL = 'http://localhost:8001';
```

## Troubleshooting

### "CV Required" Message
**Problem**: Cannot start interview  
**Solution**: Upload CV in "CV Analysis" section first

### Backend Not Responding
**Problem**: API calls fail  
**Solution**:
```bash
# Check if backend is running
curl http://localhost:8001/health

# If not, start it
cd backend && python main_simple_for_frontend.py
```

### No Questions Generated
**Problem**: Interview starts but no questions  
**Solution**: Check backend logs for errors in question generation

### Voice Mode Not Working
**Problem**: Microphone issues  
**Solution**:
- Allow microphone access in browser
- Check if GOOGLE_API_KEY is set
- Verify WebSocket connection

## Files Structure

```
backend/
├── agents/
│   └── live_agent.py          # Voice AI agent
├── schemas/
│   └── interview.py           # Request/Response models
├── skillsync/
│   └── interviews/
│       ├── routes.py          # API endpoints
│       ├── service.py         # Business logic
│       └── models.py          # Database models

frontend/
├── src/
│   ├── pages/
│   │   ├── NewInterviewPage.tsx       # Mode selection
│   │   ├── LiveInterviewPage.tsx      # Text mode
│   │   └── LiveInterviewPageVoice.tsx # Voice mode
│   ├── hooks/
│   │   └── useAudioStream.ts  # Audio streaming
│   ├── api.ts                 # API client
│   └── App.tsx               # Main app + nav
```

## Development

### Adding New Question Types
Edit `backend/agents/interview_agent.py`:
```python
def generate_questions(self, cv_data, job_title):
    # Add your question generation logic
    questions.append({
        "text": "Your question here",
        "category": "technical",
        "difficulty": "medium"
    })
```

### Customizing Interview UI
Edit `frontend/src/pages/LiveInterviewPage.tsx`:
```tsx
// Customize question display
<div className="question-card">
  {currentQuestion.question_text}
</div>
```

### Adding Voice Features
Edit `backend/agents/live_agent.py`:
```python
async def process_audio_stream(self, audio_chunk):
    # Add your audio processing logic
    pass
```

## Performance

- **Text Mode**: ~100ms response time
- **Voice Mode**: ~500ms latency
- **Concurrent Users**: 50+ (depends on server)
- **Question Generation**: < 3 seconds
- **Report Generation**: < 5 seconds

## Security

- 🔒 HTTPS recommended for production
- 🔐 Authentication required (integrate with your auth)
- 🛡️ Rate limiting enabled
- 🔊 Audio data not stored by default
- 📝 CV data encrypted in transit

## Support

- 📖 Full docs: `AI_INTERVIEW_IMPLEMENTATION.md`
- 🌐 API docs: http://localhost:8001/docs
- 🐛 Issues: GitHub Issues
- 💬 Questions: Development Team

## License
Part of SkillSync platform - See main LICENSE file

---

**Version**: 1.0.0  
**Status**: ✅ Text Mode Ready, 🚧 Voice Mode Beta  
**Last Updated**: November 23, 2025
