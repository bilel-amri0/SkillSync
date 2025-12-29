import datetime

from backend.main_enhanced import CVAnalysisResponse, _select_primary_skills


def _build_analysis(summary: str, skills):
    return CVAnalysisResponse(
        analysis_id="analysis-tests",
        skills=skills,
        hard_skills=skills,
        soft_skills=[],
        experience_years=3.0,
        job_titles=["Engineer"],
        education=["Engineering"],
        summary=summary,
        confidence_score=0.82,
        timestamp=datetime.datetime.utcnow().isoformat()
    )


def test_ai_summary_prioritizes_ai_stack():
    analysis = _build_analysis(
        "Machine learning engineer focusing on MLOps, LLM safety, and TensorFlow production.",
        ["React", "TensorFlow", "Python", "LangChain"]
    )

    prioritized = _select_primary_skills(analysis)

    assert prioritized[0].lower() == "machine learning"
    assert "TensorFlow" in prioritized[:3]
    assert prioritized.index("TensorFlow") < prioritized.index("React")


def test_web_summary_keeps_web_focus():
    analysis = _build_analysis(
        "Full stack JavaScript developer building React and Node.js platforms.",
        ["React", "Node.js", "TypeScript", "MongoDB"]
    )

    prioritized = _select_primary_skills(analysis)

    assert prioritized[0] == "React"
    assert all(skill not in prioritized for skill in ["Machine Learning", "Data Science"])
