"""
🚀 Test ML-Driven Career Guidance System
Tests the fully ML-powered career recommendations
"""
import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from datetime import datetime

console = Console()

# Sample CV for testing
SAMPLE_CV = """
BILEL AMRI
Email: bilel.amri@example.com | Phone: +1234567890 | Location: San Francisco, CA
LinkedIn: linkedin.com/in/bilelamri | GitHub: github.com/bilelamri

PROFESSIONAL SUMMARY
Experienced Software Engineer with 4 years of expertise in Python, machine learning, and backend development.
Passionate about AI/ML and building scalable systems. Strong background in data science and cloud technologies.

WORK EXPERIENCE

Machine Learning Engineer | TechCorp Inc. | 2022 - Present
- Developed and deployed 10+ ML models using TensorFlow and PyTorch
- Built data pipelines processing 5TB+ daily data using Apache Spark
- Implemented MLOps practices with Docker, Kubernetes, and CI/CD
- Improved model accuracy by 25% through feature engineering
- Collaborated with data scientists and engineers on AI products

Software Developer | StartupXYZ | 2020 - 2022
- Built RESTful APIs using FastAPI and Django
- Worked with React for frontend development
- Managed PostgreSQL databases and Redis caching
- Deployed applications on AWS (EC2, S3, Lambda)
- Implemented automated testing with pytest

EDUCATION
Bachelor of Science in Computer Science
University of California | 2016 - 2020

TECHNICAL SKILLS
- Languages: Python, JavaScript, SQL, Java
- ML/AI: TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy
- Backend: FastAPI, Django, Flask, Node.js
- Frontend: React, HTML, CSS
- DevOps: Docker, Kubernetes, CI/CD, Jenkins
- Cloud: AWS (EC2, S3, Lambda, SageMaker)
- Databases: PostgreSQL, MongoDB, Redis
- Tools: Git, Linux, Jupyter

PROJECTS
- AI Resume Analyzer: Built ML system to analyze resumes and provide career guidance
- Stock Prediction Model: LSTM-based model for stock price forecasting
- E-commerce Platform: Full-stack application with React and Django

CERTIFICATIONS
- AWS Certified Developer - Associate
- TensorFlow Developer Certificate (in progress)
"""

def test_ml_career_guidance():
    """Test the ML-driven career guidance endpoint"""
    
    console.print(Panel.fit(
        "🚀 [bold cyan]Testing ML-Driven Career Guidance System[/bold cyan]",
        border_style="cyan"
    ))
    
    # API endpoint
    url = "http://localhost:8001/api/v1/career-guidance"
    
    console.print("\n📤 [yellow]Sending CV to ML career guidance API...[/yellow]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=100)
        
        try:
            # Send request
            start_time = datetime.now()
            response = requests.post(
                url,
                json={"cv_content": SAMPLE_CV},
                headers={"Content-Type": "application/json"}
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            progress.update(task, completed=100)
            
            if response.status_code == 200:
                console.print(f"✅ [green]API Response: 200 OK[/green] ({processing_time:.2f}s)")
                
                result = response.json()
                
                # Save to file
                with open('ml_career_guidance_result.json', 'w') as f:
                    json.dump(result, f, indent=2)
                console.print("💾 [green]Saved full response to ml_career_guidance_result.json[/green]")
                
                # Display results
                display_ml_results(result)
                
            else:
                console.print(f"❌ [red]API Error: {response.status_code}[/red]")
                console.print(response.text)
                
        except Exception as e:
            console.print(f"❌ [red]Request failed: {e}[/red]")


def display_ml_results(result):
    """Display ML career guidance results in a beautiful format"""
    
    # ========== METADATA ==========
    console.print("\n" + "="*80)
    console.print("[bold magenta]📊 ML SYSTEM METADATA[/bold magenta]")
    console.print("="*80)
    
    metadata = result.get('metadata', {})
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("🤖 ML Model:", metadata.get('ml_model', 'N/A'))
    meta_table.add_row("🚀 Engine Version:", metadata.get('engine_version', 'N/A'))
    meta_table.add_row("⚡ Processing Time:", f"{metadata.get('processing_time_seconds', 0)}s")
    meta_table.add_row("📝 CV Skills:", str(metadata.get('cv_skills_count', 0)))
    meta_table.add_row("💼 Jobs Recommended:", str(metadata.get('jobs_recommended', 0)))
    meta_table.add_row("🎓 Certs Recommended:", str(metadata.get('certs_recommended', 0)))
    meta_table.add_row("🎯 Roadmap Phases:", str(metadata.get('roadmap_phases', 0)))
    console.print(meta_table)
    
    # ========== JOB RECOMMENDATIONS ==========
    console.print("\n" + "="*80)
    console.print("[bold cyan]💼 ML-POWERED JOB RECOMMENDATIONS[/bold cyan]")
    console.print("="*80)
    
    jobs = result.get('job_recommendations', [])
    if jobs:
        for i, job in enumerate(jobs[:5], 1):
            console.print(f"\n[bold yellow]#{i} {job['title']}[/bold yellow]")
            console.print(f"   🤖 ML Similarity: [cyan]{job['similarity_score']*100:.1f}%[/cyan]")
            console.print(f"   🎯 ML Confidence: [green]{job['confidence']*100:.1f}%[/green]")
            salary = job['predicted_salary']
            console.print(f"   💰 Predicted Salary: [green]${salary['min']:,} - ${salary['max']:,}[/green] (ML-computed)")
            console.print(f"   📈 Growth Potential: [magenta]{job['growth_potential']}[/magenta]")
            console.print(f"   ✅ Matching Skills ({len(job['matching_skills'])}): {', '.join(job['matching_skills'][:5])}")
            if job['skill_gaps']:
                console.print(f"   📚 Skills to Learn ({len(job['skill_gaps'])}): {', '.join(job['skill_gaps'][:5])}")
            console.print(f"   💡 ML Reasoning:")
            for reason in job['reasons'][:3]:
                console.print(f"      • {reason}")
    else:
        console.print("[red]No job recommendations found[/red]")
    
    # ========== CERTIFICATION RECOMMENDATIONS ==========
    console.print("\n" + "="*80)
    console.print("[bold cyan]🎓 ML-RANKED CERTIFICATIONS[/bold cyan]")
    console.print("="*80)
    
    certs = result.get('certification_recommendations', [])
    if certs:
        for i, cert in enumerate(certs[:5], 1):
            console.print(f"\n[bold yellow]#{i} {cert['name']}[/bold yellow]")
            console.print(f"   🤖 ML Relevance: [cyan]{cert['relevance_score']*100:.1f}%[/cyan]")
            console.print(f"   🎯 Skill Alignment: [green]{cert['skill_alignment']*100:.1f}%[/green]")
            console.print(f"   💰 Predicted ROI: [green]{cert['predicted_roi']}[/green]")
            console.print(f"   ⏱️  Estimated Time: [yellow]{cert['estimated_time']}[/yellow]")
            console.print(f"   📈 Career Boost: [magenta]{cert['career_boost']}[/magenta]")
            console.print(f"   💡 ML Reasoning:")
            for reason in cert['reasons'][:3]:
                console.print(f"      • {reason}")
    else:
        console.print("[red]No certification recommendations found[/red]")
    
    # ========== LEARNING ROADMAP ==========
    console.print("\n" + "="*80)
    console.print("[bold cyan]🎯 ML-OPTIMIZED LEARNING ROADMAP[/bold cyan]")
    console.print("="*80)
    
    roadmap = result.get('learning_roadmap', {})
    if roadmap:
        console.print(f"\n[bold]Total Duration:[/bold] {roadmap['total_duration_weeks']} weeks ({roadmap['total_duration_months']} months)")
        console.print(f"[bold]ML Success Prediction:[/bold] [green]{roadmap['predicted_success_rate']}[/green]")
        console.print(f"[bold]Personalization Score:[/bold] [cyan]{roadmap['personalization_score']}[/cyan]")
        console.print(f"[bold]Learning Strategy:[/bold] {roadmap['learning_strategy']}")
        
        phases = roadmap.get('phases', [])
        for phase in phases:
            console.print(f"\n[bold magenta]{phase['phase_name']}[/bold magenta]")
            console.print(f"   ⏱️  Duration: {phase['duration_weeks']} weeks ({phase['duration_months']} months)")
            console.print(f"   🎓 ML Success Probability: [green]{phase['success_probability']}[/green]")
            console.print(f"   💪 Effort Level: {phase['effort_level']}")
            console.print(f"   📚 Skills to Learn ({len(phase['skills_to_learn'])}): {', '.join(phase['skills_to_learn'])}")
            
            if phase['learning_resources']:
                console.print(f"   📖 ML-Curated Resources:")
                for resource in phase['learning_resources'][:3]:
                    console.print(f"      • {resource['title']} ({resource['provider']}) - {resource['duration']} ⭐ {resource['rating']}")
            
            console.print(f"   🎯 Milestones:")
            for milestone in phase['milestones'][:3]:
                console.print(f"      ✓ {milestone}")
    else:
        console.print("[red]No learning roadmap found[/red]")
    
    # ========== XAI INSIGHTS ==========
    console.print("\n" + "="*80)
    console.print("[bold cyan]🧠 EXPLAINABLE AI (XAI) INSIGHTS[/bold cyan]")
    console.print("="*80)
    
    xai = result.get('xai_insights', {})
    if xai:
        # How we analyzed
        console.print("\n[bold yellow]🔍 How We Analyzed Your CV:[/bold yellow]")
        analysis = xai.get('how_we_analyzed_your_cv', {})
        console.print(f"   Method: {analysis.get('method', 'N/A')}")
        console.print(f"   Model: {analysis.get('model', 'N/A')}")
        steps = analysis.get('steps', [])
        for step in steps[:5]:
            console.print(f"   • {step}")
        
        # Job matching
        console.print("\n[bold yellow]💼 Job Matching Explanation:[/bold yellow]")
        job_exp = xai.get('job_matching_explanation', {})
        console.print(f"   Method: {job_exp.get('method', 'N/A')}")
        how_it_works = job_exp.get('how_it_works', [])
        for step in how_it_works[:4]:
            console.print(f"   • {step}")
        
        # Cert ranking
        console.print("\n[bold yellow]🎓 Certification Ranking Explanation:[/bold yellow]")
        cert_exp = xai.get('certification_ranking_explanation', {})
        console.print(f"   Method: {cert_exp.get('method', 'N/A')}")
        how_it_works = cert_exp.get('how_it_works', [])
        for step in how_it_works[:4]:
            console.print(f"   • {step}")
        
        # Learning path
        console.print("\n[bold yellow]🎯 Learning Path Optimization:[/bold yellow]")
        learning_exp = xai.get('learning_path_explanation', {})
        console.print(f"   Method: {learning_exp.get('method', 'N/A')}")
        how_it_works = learning_exp.get('how_it_works', [])
        for step in how_it_works[:5]:
            console.print(f"   • {step}")
        
        # ML confidence
        console.print("\n[bold yellow]📊 ML Confidence Scores:[/bold yellow]")
        confidence = xai.get('ml_confidence_scores', {})
        for key, value in confidence.items():
            console.print(f"   • {key}: {value}")
        
        # Key insights
        console.print("\n[bold yellow]💡 Key Insights:[/bold yellow]")
        insights = xai.get('key_insights', [])
        for insight in insights:
            console.print(f"   • {insight}")
    else:
        console.print("[red]No XAI insights found[/red]")
    
    console.print("\n" + "="*80)
    console.print("[bold green]✅ ML CAREER GUIDANCE COMPLETE[/bold green]")
    console.print("="*80 + "\n")


if __name__ == "__main__":
    console.print("""
[bold cyan]🤖 ML-DRIVEN CAREER GUIDANCE TEST[/bold cyan]

This test demonstrates the fully ML-powered career guidance system:
✅ Semantic job matching using transformer embeddings
✅ ML-based salary predictions
✅ Intelligent certification ranking
✅ Optimized learning path generation
✅ Complete explainability (XAI)

[yellow]Make sure the backend server is running on port 8001![/yellow]
    """)
    
    input("Press Enter to start the test...")
    test_ml_career_guidance()
