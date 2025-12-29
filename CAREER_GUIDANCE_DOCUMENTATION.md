# 🎯 Career Guidance Engine with Explainable AI (XAI)

## Overview

The Career Guidance Engine is an advanced ML-powered system that provides **actionable career recommendations** with **explainable AI insights**. It analyzes CVs using the advanced ML endpoint and generates personalized guidance including job recommendations, certification suggestions, learning roadmaps, and XAI reasoning.

---

## 🚀 Features

### 1. **Job Recommendations** 💼
- **Multi-factor scoring algorithm**:
  - Required skills match (70%)
  - Preferred skills match (30%)
  - Industry relevance (+10% boost)
- **Match scores** with percentages
- **Missing skills** identified for each role
- **Salary ranges** and growth potential
- **XAI reasoning** explaining why each job was recommended

### 2. **Certification Recommendations** 🎓
- **Priority-based scoring**:
  - Missing skills coverage (3 points)
  - Industry relevance (2 points)
  - Builds on existing skills (1 point)
- **Personalized priorities** (Very High, High, Medium)
- **Cost and duration** information
- **Career impact** assessment
- **XAI reasons** for each certification

### 3. **Learning Roadmap** 🗺️
- **Step-by-step phases**:
  - Foundation Building (1-2 months)
  - Practical Application (2-3 months)
  - Specialization (2-4 months)
- **Current → Target level** progression
- **Timeline estimates** based on goals
- **Resource recommendations** for each phase
- **XAI reasoning** for roadmap structure

### 4. **Explainable AI (XAI) Insights** 🧠
- **Algorithm transparency**: Shows how recommendations were generated
- **Confidence scores**: Indicates reliability of predictions
- **ML features used**: Lists all ML models and techniques
- **Decision rationale**: Explains why each recommendation was made
- **Threshold logic**: Shows minimum requirements for recommendations

---

## 📋 API Endpoint

### **POST /api/v1/career-guidance**

**Input:**
```json
{
  "cv_content": "Your CV text here..."
}
```

**Output:**
```json
{
  "cv_analysis": {
    "skills": ["Python", "Machine Learning", "React"],
    "seniority": "Mid",
    "industries": [
      {"name": "Data_Science", "confidence": 0.85},
      {"name": "Software_Engineering", "confidence": 0.72}
    ],
    "projects": [...],
    "portfolio_links": {...},
    "experience_years": 3
  },
  "recommended_jobs": [
    {
      "title": "ML Engineer",
      "match_score": "85.3%",
      "salary_range": "$90k-$150k",
      "growth_potential": "Very High",
      "reasons": [
        "✅ All 5 required skills matched",
        "✅ 3/4 preferred skills matched",
        "✅ Strong industry fit: Data_Science, AI_ML"
      ],
      "required_skills": ["Python", "Machine Learning", "TensorFlow"],
      "missing_skills": ["Kubernetes", "MLOps"]
    }
  ],
  "recommended_certifications": [
    {
      "name": "TensorFlow Developer Certificate",
      "provider": "Google",
      "priority": "Very High",
      "duration": "2-3 months",
      "cost": "$100",
      "reasons": [
        "🎯 Covers missing skills: TensorFlow",
        "🏢 Relevant to your top industries: AI_ML, Data_Science",
        "💼 Very High - Essential for ML roles"
      ],
      "skills_gained": ["TensorFlow", "Deep Learning", "ML"],
      "career_impact": "Very High - Essential for ML roles"
    }
  ],
  "learning_roadmap": {
    "current_level": "Mid",
    "target_level": "Senior",
    "timeline": "12-18 months",
    "phases": [
      {
        "phase": "Foundation Building",
        "duration": "1-2 months",
        "priority": "Very High",
        "skills": ["Kubernetes", "MLOps", "Docker"],
        "reason": "These skills are required for ML Engineer",
        "resources": ["Online courses", "Documentation", "Projects"]
      }
    ],
    "reasoning": [
      "📊 Current Level: Mid",
      "🎯 Target Level: Senior",
      "⏱️ Estimated Timeline: 12-18 months",
      "💼 Top Target Role: ML Engineer"
    ]
  },
  "xai_insights": {
    "analysis_summary": {
      "total_skills": 15,
      "top_industries": ["Data_Science", "Software_Engineering"],
      "skill_extraction_method": "Semantic + ML (paraphrase-mpnet-base-v2 + BERT-NER)"
    },
    "job_matching_logic": {
      "algorithm": "Multi-factor scoring: Required skills (70%) + Preferred skills (30%) + Industry fit (10%)",
      "threshold": "Minimum 50% required skills match",
      "top_match": {
        "job": "ML Engineer",
        "score": "85.3%",
        "reasons": [...]
      }
    },
    "certification_logic": {
      "algorithm": "Priority scoring: Missing skills (3pts) + Industry relevance (2pts) + Builds on existing (1pt)",
      "threshold": "Minimum 2 points for recommendation",
      "rationale": "Focus on filling skill gaps for target roles"
    },
    "roadmap_logic": {
      "phases": 3,
      "methodology": "Prioritize critical missing skills → Practical application → Specialization",
      "personalization": "Tailored for Mid level targeting ML Engineer"
    },
    "confidence_scores": {
      "job_recommendations": "High",
      "certification_fit": "High",
      "roadmap_accuracy": "High - Based on industry standards"
    },
    "ml_features_used": [
      "Semantic skill extraction (paraphrase-mpnet-base-v2)",
      "Industry classification (3-class confidence)",
      "Project detection (NER + pattern matching)",
      "Portfolio analysis (GitHub, LinkedIn)",
      "Seniority prediction (ML-based)"
    ]
  }
}
```

---

## 🧪 Testing

### Quick Test

```bash
cd backend
python test_career_guidance.py
```

### Expected Output

```
🎯 CAREER GUIDANCE ENGINE TEST
================================================================================

📊 CV ANALYSIS SUMMARY
================================================================================

🎯 Seniority Level: Mid
📅 Experience: 3 years
💼 Total Skills: 20
   Top Skills: Python, Machine Learning, Deep Learning, TensorFlow, React

🏢 Top Industries:
   • Data_Science: 85.3% confidence
   • Software_Engineering: 72.1% confidence
   • AI_ML: 68.4% confidence

💼 RECOMMENDED JOBS
================================================================================

1. ML Engineer
   Match Score: 85.3%
   Salary Range: $90k-$150k
   Growth Potential: Very High
   
   Why this job?
      ✅ All 5 required skills matched
      ✅ 3/4 preferred skills matched
      ✅ Strong industry fit: Data_Science, AI_ML
      ✅ Experience requirement met (3 years)
   
   📚 Skills to learn:
      • Kubernetes
      • MLOps
      • Terraform

2. Data Scientist
   Match Score: 78.5%
   ...
```

---

## 🎯 How It Works

### 1. **CV Analysis** (Advanced ML)
```python
# Uses /api/v1/analyze-cv-advanced endpoint
parser = ProductionCVParser()
cv_result = parser.parse_cv(cv_content)

# Extracts:
# - Skills (semantic + ML)
# - Seniority level (ML-predicted)
# - Industries (3-class confidence)
# - Projects (NER + pattern matching)
# - Portfolio links (GitHub, LinkedIn)
```

### 2. **Job Matching Algorithm**
```python
# Multi-factor scoring
required_score = (matched_required / total_required) * 0.7  # 70%
preferred_score = (matched_preferred / total_preferred) * 0.3  # 30%
industry_boost = 0.1 if industry_matches else 0.0  # +10%

match_score = required_score + preferred_score + industry_boost

# Minimum threshold: 50% required skills
if required_score >= 0.35:  # 50% of 70%
    recommend_job()
```

### 3. **Certification Prioritization**
```python
# Priority scoring
priority_score = 0
priority_score += 3 if covers_missing_skills else 0
priority_score += 2 if industry_relevant else 0
priority_score += 1 if builds_on_existing else 0

# Minimum threshold: 2 points
if priority_score >= 2:
    recommend_cert()

# Priority levels
# >= 5: Very High
# >= 4: High
# >= 2: Medium
```

### 4. **Learning Roadmap Generation**
```python
# Phase 1: Foundation (1-2 months)
critical_missing = top_job.missing_skills[:3]

# Phase 2: Practical Application (2-3 months)
build_projects = ["Portfolio", "GitHub", "Open Source"]

# Phase 3: Specialization (2-4 months)
advanced_skills = top_job.missing_skills[3:6]

# Timeline calculation
timeline = calculate_based_on(current_level, target_level, missing_skills)
```

---

## 📊 Job Database

The engine includes **10+ job roles** across different industries:

### AI/ML Roles
- ML Engineer ($90k-$150k)
- Data Scientist ($80k-$140k)
- AI Research Engineer ($100k-$180k)

### Software Engineering Roles
- Full Stack Developer ($70k-$130k)
- Backend Engineer ($75k-$135k)
- Frontend Developer ($65k-$125k)

### DevOps/Cloud Roles
- DevOps Engineer ($85k-$145k)
- Cloud Architect ($110k-$180k)

### Entry Level Roles
- Junior Software Developer ($50k-$80k)
- Junior Data Analyst ($45k-$75k)

Each role includes:
- Required skills
- Preferred skills
- Minimum experience
- Target industries
- Salary range
- Growth potential
- Seniority levels

---

## 🎓 Certification Database

The engine recommends from **10+ certifications**:

### Cloud Certifications
- AWS Certified Solutions Architect ($150, 3-4 months)
- Google Cloud Professional Data Engineer ($200, 2-3 months)
- Microsoft Azure Fundamentals ($99, 1-2 months)

### AI/ML Certifications
- TensorFlow Developer Certificate ($100, 2-3 months)
- Deep Learning Specialization ($49/month, 3-4 months)
- AWS Machine Learning Specialty ($300, 2-3 months)

### DevOps Certifications
- Certified Kubernetes Administrator ($395, 2-3 months)
- Docker Certified Associate ($195, 1-2 months)

### Development Certifications
- Meta Front-End Developer Professional ($49/month, 5-6 months)
- Python for Everybody Specialization ($49/month, 3-4 months)

---

## 🧠 XAI (Explainable AI) Features

### Why XAI Matters
- **Transparency**: Users understand why recommendations were made
- **Trust**: Clear reasoning builds confidence in the system
- **Actionability**: Users know exactly what to do next
- **Debugging**: Easy to identify and fix recommendation issues

### XAI Components

1. **Analysis Summary**
   - Shows total skills detected
   - Lists top industries with confidence
   - Explains skill extraction method

2. **Job Matching Logic**
   - Reveals scoring algorithm
   - Shows minimum thresholds
   - Explains top match reasoning

3. **Certification Logic**
   - Shows priority scoring formula
   - Explains recommendation criteria
   - Clarifies rationale

4. **Roadmap Logic**
   - Explains phase structure
   - Shows methodology
   - Describes personalization

5. **Confidence Scores**
   - Job recommendations confidence
   - Certification fit confidence
   - Roadmap accuracy confidence

6. **ML Features Used**
   - Lists all ML models
   - Shows techniques applied
   - Indicates data sources

---

## 🔧 Integration with Frontend

### Frontend API Call
```typescript
// api.ts
export const getCareerGuidance = async (cvContent: string) => {
  const response = await apiClient.post('/api/v1/career-guidance', {
    cv_content: cvContent
  });
  return response.data;
};

// Component usage
const guidance = await getCareerGuidance(cvText);

// Display recommendations
guidance.recommended_jobs.forEach(job => {
  console.log(`${job.title}: ${job.match_score}`);
  job.reasons.forEach(reason => console.log(`  ${reason}`));
});
```

---

## 📈 Performance

- **First analysis**: ~20-25 seconds (loads ML models)
- **Subsequent analyses**: ~500ms-1s (models cached)
- **Job recommendations**: Generated in <100ms
- **Certification recommendations**: Generated in <50ms
- **Learning roadmap**: Generated in <50ms
- **XAI insights**: Generated in <10ms

---

## 🎯 Use Cases

### 1. **Job Seekers**
- Get personalized job recommendations based on skills
- Understand exactly what skills to learn
- Plan career progression with clear roadmap

### 2. **Career Changers**
- Identify transferable skills
- Find suitable transition roles
- Get certification recommendations for new field

### 3. **Students**
- Discover career paths matching their skills
- Build portfolio with guided projects
- Plan learning path for target roles

### 4. **Recruiters**
- Match candidates to roles objectively
- Identify skill gaps quickly
- Provide actionable feedback to candidates

---

## 🚀 Future Enhancements

### Phase 1 (Current) ✅
- [x] Job recommendations with XAI
- [x] Certification recommendations
- [x] Learning roadmap generation
- [x] Multi-factor scoring algorithm
- [x] Priority-based cert selection

### Phase 2 (Planned)
- [ ] Salary prediction based on skills
- [ ] Location-based job recommendations
- [ ] Company culture fit analysis
- [ ] Interview preparation guidance
- [ ] Skill gap visualization

### Phase 3 (Future)
- [ ] Real-time job market data integration
- [ ] Personalized learning content recommendations
- [ ] Career trajectory simulation
- [ ] Mentor matching
- [ ] Success rate tracking

---

## 📝 Example JSON Output

See `career_guidance_result.json` after running the test for a complete example.

---

## 🤝 Contributing

To add new job roles or certifications:

1. Edit `career_guidance_engine.py`
2. Add to `_load_job_database()` or `_load_certification_database()`
3. Follow existing format
4. Test with `test_career_guidance.py`

---

## 📞 Support

- **API Docs**: http://localhost:8001/docs
- **Test Script**: `python test_career_guidance.py`
- **Log Output**: Check terminal for detailed XAI reasoning

---

**Status**: 🟢 Production Ready
**Version**: 1.0.0
**Last Updated**: November 24, 2025
