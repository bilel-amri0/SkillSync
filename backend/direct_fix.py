"""
Direct fix for the corrupted section in main_simple_for_frontend.py
This reads the file, finds line 658, and replaces the corrupted section
"""

# Read the entire file
with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with month 9 (line 658-659 approximately)
for i, line in enumerate(lines):
    if '"month": 9,' in line and i > 650 and i < 665:
        print(f"Found corrupted section at line {i+1}")
        
        # Replace from this line until the module try: block
        # Complete the dictionary properly
        completion = '''                    "month": 9,
                    "title": "Cross-Domain Growth",
                    "description": "Expand into adjacent technologies",
                    "focus_areas": ["Technology Breadth", "System Design"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Senior Readiness",
                    "description": "Prepare for senior role",
                    "focus_areas": ["Leadership", "Impact"],
                    "status": "upcoming"
                }
            ]
        
        career_roadmap_data = {
            "target_role": f"Senior {role_focus.title()} Developer" if cv_analysis['experience_level'] != 'senior' else f"Lead {role_focus.title()} Architect",
            "timeline_months": 12,
            "milestones": roadmap_milestones,
            "current_level": cv_analysis['experience_level'],
            "focus_domains": cv_analysis['primary_domains'],
            "personalized": True
        }
        
        formatted_recommendations["CAREER_ROADMAP"] = career_roadmap_data
        
        return formatted_recommendations

try:
    import sys
    import os
    workspace_path = '/workspace'
    if workspace_path not in sys.path:
        sys.path.append(workspace_path)
    
    if not detect_ml_configuration():
        logger.info(" Mode fallback: recommandations bases sur des rgles")
    
'''
        
        # Find where to stop (find the next if False and not ML_MODE_ENABLED)
        end_idx = i + 1
        for j in range(i+1, min(i+20, len(lines))):
            if 'if False and not ML_MODE_ENABLED:' in lines[j]:
                end_idx = j
                break
        
        # Replace the corrupted section
        lines[i] = completion
        # Remove the corrupted lines
        for _ in range(end_idx - i - 1):
            lines.pop(i+1)
        
        break

# Write the file back
with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(" File fixed successfully!")
