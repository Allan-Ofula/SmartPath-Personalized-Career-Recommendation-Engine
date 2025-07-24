# --- Required Libraries --- 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned job profiles data
job_profiles_clean = pd.read_csv("data/job_profiles_clean.csv")

# Hybrid recommender function: RIASEC + Education + Skills (Kenya-Job Market optimized Recommender)
def hybrid_similarity_recommender(user_profile, riasec_weight=0.4, skill_weight=0.5, edu_weight=0.1):
    global job_profiles_clean

    job_df = job_profiles_clean.copy()

    # --- RIASEC similarity ---
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']
    user_vector = np.array([user_profile.get(code, 0) for code in riasec_cols]).reshape(1, -1)
    job_vectors = job_df[riasec_cols].values
    job_df['User RIASEC Similarity'] = cosine_similarity(user_vector, job_vectors)[0]

    # --- Education normalization ---
    edu_mapping = {
        "Less than High School": 1,
        "High School Diploma or Equivalent": 2,
        "Some College Courses": 3,
        "Associate Degree": 4,
        "Bachelor's Degree": 5,
        "Master's Degree": 6,
        "Doctoral or Professional Degree": 7,
        "Post-Doctoral Training": 8
    }

    # Normalize inconsistent education labels in dataset
    label_normalization = {
        "Less than High School": "Less than High School",
        "High School Diploma or equivalent": "High School Diploma or Equivalent",
        "Post-Secondary Certificate": "Some College Courses",
        "Associate's Degree": "Associate Degree",
        "Bachelor's Degree": "Bachelor's Degree",
        "Post-Baccalaureate Certificate": "Bachelor's Degree",
        "Post-Master's Certificate": "Master's Degree",
        "First Professional Degree": "Master's Degree",
        "Master's Degree": "Master's Degree",
        "Doctoral Degree": "Doctoral or Professional Degree",
        "Post-Doctoral Training": "Post-Doctoral Training"
    }

    # Apply normalization
    job_df['Normalized Education Category'] = job_df['Education Category Label'].map(label_normalization)
    job_df['Education Numeric'] = job_df['Normalized Education Category'].map(edu_mapping).fillna(1)

    # Calculate normalized score
    max_edu_score = max(edu_mapping.values())
    user_edu_score = edu_mapping.get(user_profile.get('education_level'), 1)
    job_df['Normalized Education Score'] = job_df['Education Numeric'] / max_edu_score

    # 🔴 FILTER: Only jobs the user qualifies for (education level)
    job_df = job_df[job_df['Education Numeric'] <= user_edu_score]

    # --- Skill similarity ---
    skill_cols = [col for col in job_df.columns if col.startswith("Skill List_")]
    user_skills = user_profile.get('skills', [])
    user_skill_vector = np.zeros((1, len(skill_cols)))

    for i, skill in enumerate(skill_cols):
        skill_name = skill.replace("Skill List_", "").lower()
        if any(skill_name in s.lower() for s in user_skills):
            user_skill_vector[0][i] = 1

    job_skill_matrix = job_df[skill_cols].fillna(0).values
    job_df['User Skill Similarity'] = cosine_similarity(user_skill_vector, job_skill_matrix)[0]

    # --- Final Hybrid weighted score ---
    job_df['Hybrid Recommendation Score'] = (
        (riasec_weight * job_df['User RIASEC Similarity']) +
        (skill_weight * job_df['User Skill Similarity']) +
        (edu_weight * job_df['Normalized Education Score'])
    )

    # --- Top 10 matches ---
    top_matches = job_df.sort_values('Hybrid Recommendation Score', ascending=False).head(10)

    # --- Personalized Message (for UI) ----
    user_name = user_profile.get('user_name', 'User')
    personalized_message = f"Hi {user_name}, below are the careers that match your RIASEC scores, skills, and education level."

    # --- Return Output ---
    return (
        top_matches[[ 
            'Title', 'Description', 'Education Level', 'Preparation Level',
            'Education Category Label', 'Hybrid Recommendation Score',
            'User RIASEC Similarity', 'Normalized Education Score', 'User Skill Similarity',
            'R', 'I', 'A', 'S', 'E', 'C'
        ]],
        {
            "personalized_message": personalized_message,
            "weights_used": {
                "RIASEC Weight": riasec_weight,
                "Skill Weight": skill_weight,
                "Education Weight": edu_weight
            }
        }
    )
