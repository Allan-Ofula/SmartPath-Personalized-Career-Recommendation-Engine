# recommender_engine.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_recommendations(user_profile, top_n=10):
    """
    Generate top N job recommendations based on user's RIASEC scores,
    normalized education level (0–1), and selected skill indicators.
    """

    # --- Load job dataset ---
    job_profiles_clean = pd.read_csv("data/job_profiles_clean.csv").copy()

    # --- Extract user RIASEC and normalize ---
    riasec_traits = ['R', 'I', 'A', 'S', 'E', 'C']
    user_riasec = np.array([
        user_profile['R'], user_profile['I'], user_profile['A'],
        user_profile['S'], user_profile['E'], user_profile['C']
    ])
    if user_riasec.sum() > 0:
        user_riasec = user_riasec / user_riasec.sum()

    # --- Compute RIASEC Similarity ---
    job_vectors = job_profiles_clean[riasec_traits].values
    riasec_similarities = cosine_similarity([user_riasec], job_vectors)[0]
    job_profiles_clean['User RIASEC Similarity'] = riasec_similarities

    # --- Education Similarity ---
    user_edu_score = user_profile.get("education_level", 0)
    if "Normalized Education Score" not in job_profiles_clean.columns:
        raise KeyError("Missing 'Normalized Education Score' in dataset")
    
    job_profiles_clean['Education Similarity'] = 1 - abs(
        job_profiles_clean['Normalized Education Score'] - user_edu_score
    )

    # Filter out over-qualified jobs
    job_profiles_clean = job_profiles_clean[
        job_profiles_clean["Normalized Education Score"] <= user_edu_score + 0.01
    ]
    if job_profiles_clean.empty:
        return pd.DataFrame(), {"num_recommendations": 0}

    # --- Skill Similarity ---
    skill_cols = [col for col in job_profiles_clean.columns if col.startswith("Skill List_")]
    user_skills = user_profile.get('skills', [])
    
    user_skill_vector = np.zeros((1, len(skill_cols)))
    for i, skill_col in enumerate(skill_cols):
        if skill_col in user_skills:
            user_skill_vector[0, i] = 1

    job_skill_matrix = job_profiles_clean[skill_cols].fillna(0).values
    if np.count_nonzero(user_skill_vector) > 0:
        skill_similarities = cosine_similarity(user_skill_vector, job_skill_matrix)[0]
    else:
        skill_similarities = np.zeros(len(job_profiles_clean))

    job_profiles_clean['User Skill Similarity'] = skill_similarities

    # --- Final Hybrid Score ---
    job_profiles_clean["Hybrid Recommendation Score"] = (
        0.4 * job_profiles_clean["User RIASEC Similarity"] +
        0.3 * job_profiles_clean["Education Similarity"] +
        0.3 * job_profiles_clean["User Skill Similarity"]
    )

    # --- Return top N jobs ---
    top_matches = job_profiles_clean.sort_values("Hybrid Recommendation Score", ascending=False).head(top_n).copy()
    top_matches.reset_index(drop=True, inplace=True)

    return top_matches[[
        'Title', 'Description', 'Education Level', 'Preparation Level',
        'Education Category Label', 'Normalized Education Score',
        'Hybrid Recommendation Score', 'User RIASEC Similarity',
        'Education Similarity', 'User Skill Similarity',
        'R', 'I', 'A', 'S', 'E', 'C'  # Include individual RIASEC scores
    ]], {"num_recommendations": len(top_matches)}
