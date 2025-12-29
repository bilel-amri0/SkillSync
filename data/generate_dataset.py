import pandas as pd
import numpy as np
from faker import Faker
import random
import json

# --- Configuration ---
NUM_USERS = 75
NUM_RESOURCES_PER_SKILL = 10
NUM_ROADMAPS = NUM_USERS * 2 # Generate two roadmaps per user for more history

# Define core data elements for consistency
CORE_SKILLS = {
    "Programming": ["Python", "JavaScript", "SQL", "Java", "C++"],
    "Data": ["Data Analysis", "Machine Learning", "Deep Learning", "Data Visualization", "Big Data"],
    "Web Dev": ["React", "Node.js", "HTML/CSS", "Cloud (AWS/Azure)", "DevOps"],
    "Soft Skills": ["Communication", "Project Management", "Leadership", "Problem Solving"]
}
ALL_SKILLS = [skill for sublist in CORE_SKILLS.values() for skill in sublist]

ROLES = {
    "Data Scientist": ["Python", "Machine Learning", "Deep Learning", "SQL", "Big Data", "Communication"],
    "Full-Stack Developer": ["JavaScript", "React", "Node.js", "HTML/CSS", "SQL", "Cloud (AWS/Azure)", "DevOps"],
    "Software Engineer (Backend)": ["Java", "SQL", "Cloud (AWS/Azure)", "DevOps", "Problem Solving"],
    "Data Analyst": ["Python", "Data Analysis", "SQL", "Data Visualization", "Communication"],
    "Project Manager": ["Project Management", "Communication", "Leadership", "Problem Solving"]
}

INDUSTRIES = ["Tech", "Finance", "Healthcare", "E-commerce", "Consulting"]
SENIORITY = ["Junior", "Mid-level", "Senior"]
RESOURCE_TYPES = ["video", "article", "exercise"]

fake = Faker()

# --- Helper Functions ---

def generate_cv(target_role):
    """Generates a CV (skills, experience, seniority, industry) consistent with a target role."""
    required_skills = ROLES[target_role]
    
    # Determine seniority and experience
    seniority = random.choice(SENIORITY)
    if seniority == "Junior":
        experience = random.randint(0, 2)
    elif seniority == "Mid-level":
        experience = random.randint(3, 7)
    else:
        experience = random.randint(8, 20)
        
    # Determine current skills (subset of required skills)
    num_current_skills = random.randint(max(0, len(required_skills) - 5), len(required_skills) - 1)
    current_skills = random.sample(required_skills, num_current_skills)
    
    industry = random.choice(INDUSTRIES)
    
    cv = {
        "skills": current_skills,
        "experience_years": experience,
        "seniority": seniority,
        "industry": industry
    }
    
    skill_gaps = sorted(list(set(required_skills) - set(current_skills)))
    
    return cv, skill_gaps

def generate_user_profiles():
    """Generates the User Profiles dataset."""
    users = []
    for i in range(1, NUM_USERS + 1):
        user_id = f"U{i:03d}"
        name = fake.name()
        target_role = random.choice(list(ROLES.keys()))
        cv, skill_gaps = generate_cv(target_role)
        
        # Weekly hours available (realistic distribution)
        weekly_hours_available = np.random.choice([5, 10, 15, 20], p=[0.3, 0.4, 0.2, 0.1])
        
        # Combine CV into a single string/JSON for the final output
        cv_str = json.dumps(cv)
        
        users.append({
            "user_id": user_id,
            "name": name,
            "CV": cv_str,
            "target_roles": target_role,
            "weekly_hours_available": weekly_hours_available,
            "skill_gaps": skill_gaps
        })
        
    return pd.DataFrame(users)

def generate_resource_metadata():
    """Generates the Resource Metadata dataset."""
    resources = []
    resource_id_counter = 1
    
    for skill in ALL_SKILLS:
        # Determine base difficulty for the skill
        if skill in ["Deep Learning", "Big Data", "C++"]:
            base_difficulty = 4
        elif skill in ["Python", "JavaScript", "SQL", "Java", "Data Analysis", "Machine Learning", "Data Visualization", "Cloud (AWS/Azure)", "DevOps"]:
            base_difficulty = 3
        elif skill in ["React", "Node.js", "HTML/CSS"]:
            base_difficulty = 2
        else: # Soft Skills
            base_difficulty = 1
            
        for i in range(NUM_RESOURCES_PER_SKILL):
            resource_id = f"R{resource_id_counter:04d}"
            resource_type = random.choice(RESOURCE_TYPES)
            
            # Difficulty is a slight variation of the base difficulty
            difficulty = max(1, min(5, base_difficulty + random.choice([-1, 0, 1])))
            
            # Duration and Cost are correlated with difficulty and type
            if resource_type == "video":
                duration_hours = round(np.random.normal(1.5 * difficulty, 0.5), 1)
                cost = round(np.random.normal(10 * difficulty, 5), 2)
            elif resource_type == "article":
                duration_hours = round(np.random.normal(0.5 * difficulty, 0.2), 1)
                cost = round(np.random.normal(2 * difficulty, 1), 2)
            else: # exercise
                duration_hours = round(np.random.normal(1.0 * difficulty, 0.3), 1)
                cost = round(np.random.normal(5 * difficulty, 3), 2)
            
            duration_hours = max(0.1, duration_hours)
            cost = max(0.0, cost)
            
            # Past success rate is inversely correlated with difficulty
            # Harder resources have lower success rates
            base_success_rate = 100 - (difficulty * 15)
            past_success_rate = max(10, min(100, int(np.random.normal(base_success_rate, 10))))
            
            resources.append({
                "resource_id": resource_id,
                "skill": skill,
                "type": resource_type,
                "duration_hours": duration_hours,
                "cost": cost,
                "past_success_rate": past_success_rate,
                "difficulty": difficulty
            })
            resource_id_counter += 1
            
    return pd.DataFrame(resources)

def generate_roadmap_history(user_df, resource_df):
    """Generates the Roadmap History dataset with internal consistency."""
    roadmaps = []
    roadmap_id_counter = 1
    
    # Pre-calculate average resource duration and difficulty per skill
    skill_metrics = resource_df.groupby("skill").agg(
        avg_duration=('duration_hours', 'mean'),
        avg_difficulty=('difficulty', 'mean')
    ).reset_index()
    
    skill_metrics_dict = skill_metrics.set_index('skill').to_dict('index')
    
    for _, user in user_df.iterrows():
        user_id = user["user_id"]
        skill_gaps = user["skill_gaps"]
        weekly_hours = user["weekly_hours_available"]
        
        # Generate multiple roadmaps per user
        for _ in range(random.randint(1, 3)): # 1 to 3 roadmaps per user
            if not skill_gaps:
                continue # Skip if no skill gaps
                
            roadmap_id = f"RM{roadmap_id_counter:04d}"
            
            # Select a sequence of skills to learn (subset of skill gaps)
            num_skills_in_roadmap = random.randint(1, len(skill_gaps))
            skills_sequence = random.sample(skill_gaps, num_skills_in_roadmap)
            
            # Calculate planned duration
            planned_duration_hours = 0
            for skill in skills_sequence:
                # Select a random set of resources for the skill
                skill_resources = resource_df[resource_df["skill"] == skill]
                if skill_resources.empty: continue
                
                # Simple selection: 1 video, 1 article, 1 exercise
                selected_resources = []
                for r_type in RESOURCE_TYPES:
                    filtered_resources = skill_resources[skill_resources["type"] == r_type]
                    if not filtered_resources.empty:
                        r = filtered_resources.sample(1)
                        selected_resources.append(r.iloc[0])
                
                # Planned duration for the skill is the sum of selected resource durations
                skill_planned_duration = sum(r["duration_hours"] for r in selected_resources)
                planned_duration_hours += skill_planned_duration
            
            # --- Actual Duration and Completion Logic ---
            
            # Base factor for actual duration (e.g., 1.0 means actual = planned)
            # Influenced by average difficulty of the roadmap
            avg_roadmap_difficulty = np.mean([skill_metrics_dict.get(s, {}).get('avg_difficulty', 3) for s in skills_sequence])
            
            # Higher difficulty -> higher chance of taking longer
            difficulty_factor = 1 + (avg_roadmap_difficulty - 3) * 0.1
            
            # User's weekly hours available also influences completion time
            # Lower hours -> higher chance of taking longer (or dropping)
            hours_factor = 1 + (20 - weekly_hours) * 0.02
            
            # Random noise
            noise_factor = np.random.normal(1.0, 0.15)
            
            actual_duration_factor = difficulty_factor * hours_factor * noise_factor
            actual_duration_hours = round(planned_duration_hours * actual_duration_factor, 2)
            
            # --- Dropped Skills and Quiz Scores ---
            
            dropped_skills = []
            quiz_scores = {}
            
            # Probability of dropping a skill is higher if actual duration is much longer than planned
            drop_prob_base = max(0, (actual_duration_hours / planned_duration_hours) - 1.2) * 0.3
            
            for skill in skills_sequence:
                if random.random() < drop_prob_base:
                    dropped_skills.append(skill)
                    quiz_scores[skill] = None # No score if dropped
                else:
                    # Quiz score is inversely correlated with actual duration factor and difficulty
                    avg_skill_difficulty = skill_metrics_dict.get(skill, {}).get('avg_difficulty', 3)
                    
                    # Base score (e.g., 85)
                    base_score = 85
                    
                    # Penalty for taking longer
                    duration_penalty = max(0, (actual_duration_factor - 1.0) * 10)
                    
                    # Penalty for difficulty
                    difficulty_penalty = (avg_skill_difficulty - 3) * 3
                    
                    final_score = base_score - duration_penalty - difficulty_penalty + np.random.normal(0, 5)
                    quiz_scores[skill] = max(50, min(100, int(final_score)))
            
            # --- Satisfaction Score ---
            
            # Satisfaction is correlated with quiz scores and completion time
            completion_ratio = actual_duration_hours / planned_duration_hours
            
            if dropped_skills:
                satisfaction_score = random.randint(1, 2) # Low satisfaction if skills were dropped
            elif completion_ratio < 0.9:
                satisfaction_score = random.randint(4, 5) # High satisfaction if completed early
            elif completion_ratio < 1.1:
                satisfaction_score = random.randint(3, 4) # Medium/High satisfaction if on time
            else:
                satisfaction_score = random.randint(2, 3) # Medium/Low satisfaction if completed late
                
            # Final check for completion
            if len(dropped_skills) == len(skills_sequence):
                # Entire roadmap dropped
                actual_duration_hours = None
                satisfaction_score = 1
            
            roadmaps.append({
                "roadmap_id": roadmap_id,
                "user_id": user_id,
                "skills_sequence": skills_sequence,
                "planned_duration_hours": round(planned_duration_hours, 2),
                "actual_duration_hours": actual_duration_hours,
                "dropped_skills": dropped_skills,
                "quiz_scores": json.dumps(quiz_scores),
                "satisfaction_score": satisfaction_score
            })
            roadmap_id_counter += 1
            
    return pd.DataFrame(roadmaps)

# --- Generation and Export ---

def generate_and_export():
    # 1. Generate User Profiles
    user_df = generate_user_profiles()
    
    # 2. Generate Resource Metadata
    resource_df = generate_resource_metadata()
    
    # 3. Generate Roadmap History
    roadmap_df = generate_roadmap_history(user_df, resource_df)
    
    # --- Add Comments/Explanations ---
    
    # User Profiles Explanation
    user_comments = """
    # User Profiles Dataset Comments
    # - user_id: Unique identifier for the user.
    # - name: Synthetic name generated by Faker.
    # - CV: JSON string containing 'skills' (current skills), 'experience_years', 'seniority', and 'industry'.
    # - target_roles: The desired role the user is learning towards.
    # - weekly_hours_available: Self-reported hours per week available for learning. Used to influence 'actual_duration_hours' in Roadmap History.
    # - skill_gaps: List of skills required for 'target_roles' that are missing from the user's current 'CV' skills. This is the basis for roadmaps.
    """
    
    # Resource Metadata Explanation
    resource_comments = """
    # Resource Metadata Dataset Comments
    # - resource_id: Unique identifier for the learning resource.
    # - skill: The specific skill the resource teaches.
    # - type: The format of the resource (video, article, exercise).
    # - duration_hours: Estimated time to complete the resource. Correlated with 'difficulty'.
    # - cost: Monetary cost of the resource. Correlated with 'difficulty' and 'type'.
    # - past_success_rate: Historical completion rate for this resource (0-100%). Inversely correlated with 'difficulty'.
    # - difficulty: Subjective difficulty rating (1-5).
    """
    
    # Roadmap History Explanation
    roadmap_comments = """
    # Roadmap History Dataset Comments
    # - roadmap_id: Unique identifier for the learning roadmap instance.
    # - user_id: Foreign key linking to the User Profiles dataset.
    # - skills_sequence: The ordered list of skills the user planned to learn in this roadmap. A subset of the user's 'skill_gaps'.
    # - planned_duration_hours: The sum of estimated resource durations for all skills in the sequence.
    # - actual_duration_hours: The actual time taken. Influenced by 'planned_duration_hours', resource difficulty, and user's 'weekly_hours_available'. Can be None if the entire roadmap was dropped.
    # - dropped_skills: List of skills in the sequence that the user did not complete.
    # - quiz_scores: JSON string mapping skills to final quiz scores (50-100). None if the skill was dropped. Correlated with completion time and difficulty.
    # - satisfaction_score: User's self-reported satisfaction (1-5). Correlated with completion time and dropped skills.
    #
    # ML Use Cases:
    # - Regression: Predict 'actual_duration_hours' or 'quiz_scores'.
    # - Classification: Predict probability of 'roadmap completion' (based on dropped_skills being empty).
    # - Ranking: Rank resources (using Resource Metadata) based on predicted 'past_success_rate' and 'difficulty' for a given user profile.
    """
    
    # --- Export to CSV and JSON ---
    
    # CSV Export
    user_df.to_csv("user_profiles.csv", index=False)
    resource_df.to_csv("resource_metadata.csv", index=False)
    roadmap_df.to_csv("roadmap_history.csv", index=False)
    
    # JSON Export (using records orientation for easier list processing)
    user_df.to_json("user_profiles.json", orient="records", indent=4)
    resource_df.to_json("resource_metadata.json", orient="records", indent=4)
    roadmap_df.to_json("roadmap_history.json", orient="records", indent=4)
    
    # Save comments
    with open("user_profiles_comments.txt", "w") as f:
        f.write(user_comments)
    with open("resource_metadata_comments.txt", "w") as f:
        f.write(resource_comments)
    with open("roadmap_history_comments.txt", "w") as f:
        f.write(roadmap_comments)

    print("Datasets generated and saved as CSV and JSON files.")

if __name__ == "__main__":
    generate_and_export()
