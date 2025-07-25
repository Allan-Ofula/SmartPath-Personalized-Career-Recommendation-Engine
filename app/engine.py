# --- Required Libraries --- 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import traceback
import os

# Load cleaned job profiles data
job_profiles_clean = pd.read_csv("data/job_profiles_clean.csv")

def get_encoded_skill_columns():
    # Use absolute path from project root
    skills_path = Path(__file__).resolve().parent.parent / "data" / "Skills.xlsx"

    if not skills_path.exists():
        print("❌ Skills.xlsx file not found.")
        return []

    skills_df = pd.read_excel(skills_path)

    # Filter skills with 'Importance' scale
    important_skills_df = skills_df[skills_df["Scale ID"] == "IM"]
    encoded_columns = sorted(important_skills_df["Element Name"].unique().tolist())

    return encoded_columns

# --- START FUNCTION ---
def hybrid_similarity_recommender(user_profile, riasec_weight=0.4, skill_weight=0.5, edu_weight=0.1):
    global job_profiles_clean
    job_df = job_profiles_clean.copy()

    # --- 1. RIASEC SIMILARITY ---
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']
    user_vector = np.array([user_profile.get(code, 0) for code in riasec_cols]).reshape(1, -1)
    job_vectors = job_df[riasec_cols].values
    job_df['User RIASEC Similarity'] = cosine_similarity(user_vector, job_vectors)[0]

    # --- 2. EDUCATION NORMALIZATION ---
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

    # Place label_normalization dictionary here
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

    # Normalize job education levels
    job_df['Normalized Education Category'] = job_df['Education Category Label'].map(label_normalization)
    job_df['Education Numeric'] = job_df['Normalized Education Category'].map(edu_mapping).fillna(1)

    max_edu_score = max(edu_mapping.values())
    user_edu_score = edu_mapping.get(user_profile.get('education_level'), 1)
    job_df['Normalized Education Score'] = job_df['Education Numeric'] / max_edu_score

    #  FILTER JOBS THE USER IS QUALIFIED FOR:
    job_df = job_df[job_df['Education Numeric'] <= user_edu_score]

    # --- 3. SKILL SIMILARITY ---
    user_selected_skills = user_profile.get("skills", [])

    # Ensure case-insensitive comparison
    skill_cols = [col for col in job_profiles_clean.columns if col.startswith("Skill List_")]
    user_skill_vector = np.array([
        1.0 if col.replace("Skill List_", "").lower() in [s.lower() for s in user_selected_skills] else 0.0
        for col in skill_cols
    ]).reshape(1, -1)

    job_skill_matrix = job_df[skill_cols].fillna(0).values

    # Avoid NaN by masking jobs with 0 total skills
    nonzero_job_mask = job_skill_matrix.sum(axis=1) != 0
    skill_similarities = np.zeros(len(job_df))

    if np.any(user_skill_vector):  # If user selected any skills
        skill_similarities[nonzero_job_mask] = cosine_similarity(
            job_skill_matrix[nonzero_job_mask], user_skill_vector
        ).flatten()

    job_df["User Skill Similarity"] = skill_similarities

    # --- 4. FINAL HYBRID SCORE ---
    job_df['Hybrid Recommendation Score'] = (
        (riasec_weight * job_df['User RIASEC Similarity']) +
        (skill_weight * job_df['User Skill Similarity']) +
        (edu_weight * job_df['Normalized Education Score'])
    )

    # --- 5. TOP MATCHES ---
    top_matches = job_df.sort_values('Hybrid Recommendation Score', ascending=False).head(10)

    # --- 6. PERSONALIZED UI MESSAGE ---
    user_name = user_profile.get('user_name', 'User')
    personalized_message = f"Hi {user_name}, below are the careers that match your RIASEC scores, skills, and education level."

    # --- 7. FILTER OUT NON-MATCHING SKILL JOBS (optional) ---
    if user_selected_skills:
        top_matches = top_matches[top_matches['User Skill Similarity'] > 0]

    # --- 8. FALLBACK (NO SKILLS & LOW EDUCATION) ---
    if not user_selected_skills and user_profile.get("education_level") in [None, "", "Less than High School"]:
        fallback_matches = job_profiles_clean.copy()

        fallback_matches['Fallback Score'] = fallback_matches[riasec_cols].apply(
            lambda row: np.dot(user_vector.flatten(), row.values), axis=1
        )

        fallback_matches = fallback_matches.sort_values('Fallback Score', ascending=False).head(10)

        personalized_message = (
            f"Hi {user_name}, based on your interests alone, here are job suggestions you may explore. "
            "You currently have no recorded education or skills, but don’t worry – it’s a great starting point!"
        )

        # Return fallback
        return (
            fallback_matches[['Title', 'Description', 'Education Level', 'Preparation Level',
                              'Education Category Label', 'Fallback Score',
                              'R', 'I', 'A', 'S', 'E', 'C']],
            {
                "personalized_message": personalized_message,
                "weights_used": {
                    "RIASEC Weight": riasec_weight,
                    "Skill Weight": skill_weight,
                    "Education Weight": edu_weight
                }
            }
        )

    # --- FINAL RETURN ---
    return (
        top_matches[['Title', 'Description', 'Education Level', 'Preparation Level',
                     'Education Category Label', 'Hybrid Recommendation Score',
                     'User RIASEC Similarity', 'Normalized Education Score', 'User Skill Similarity',
                     'R', 'I', 'A', 'S', 'E', 'C']],
        {
            "personalized_message": personalized_message,
            "weights_used": {
                "RIASEC Weight": riasec_weight,
                "Skill Weight": skill_weight,
                "Education Weight": edu_weight
            }
        }
    )
