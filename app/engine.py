# engine.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load job profiles with RIASEC and skill columns
job_profiles_clean = pd.read_csv("data/job_profiles_clean.csv")

# --- 1. Load and Prepare Skills Data ---
def load_and_prepare_skills_data():
    import sys
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()

    skills_path = base_path / "data" / "Skills.xlsx"
    skills_df = pd.read_excel(skills_path)

    # Filter to only relevant skill rows
    if 'Element Name' not in skills_df.columns or skills_df['Element Name'].isna().all():
        raise ValueError("Skills.xlsx doesn't contain usable 'Element Name' data.")

    relevant_skills = skills_df[skills_df['Element Name'].notna()]

    # Pivot to wide format
    skills_wide = relevant_skills.pivot_table(index="ONET_Code", columns="Element Name", values="Data Value", fill_value=0)

    # Rename columns to avoid name clashes
    skills_wide = skills_wide.rename(columns=lambda x: f"Skill List_{x}" if x != "ONET_Code" else x)

    return skills_wide.reset_index()

# --- 2. Merge Skills with Job Profiles ---
def merge_skills_into_profiles():
    skills_data = load_and_prepare_skills_data()
    merged = pd.merge(job_profiles_clean, skills_data, how="left", left_on="ONET_Code_x", right_on="ONET_Code")
    return merged

# --- 3. Extract Skill Columns Dynamically ---
def get_encoded_skill_columns():
    merged_df = merge_skills_into_profiles()
    encoded_columns = [col for col in merged_df.columns if col.startswith("Skill List_")]
    return [col.replace("Skill List_", "").strip() for col in encoded_columns]

# --- 4. Hybrid Recommender ---
def hybrid_similarity_recommender(user_profile, top_n=10):
    df = merge_skills_into_profiles()

    # Extract skill columns
    skill_cols = [col for col in df.columns if col.startswith("Skill List_")]
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']

    # Prepare vectors
    df_riasec = df[riasec_cols].fillna(0).values
    user_riasec = np.array([user_profile.get(code, 0) for code in riasec_cols]).reshape(1, -1)

    riasec_similarity = cosine_similarity(df_riasec, user_riasec).flatten()

    if user_profile["skills"]:
        # Match only skills in the encoded columns
        selected_skills = [f"Skill List_{skill}" for skill in user_profile["skills"] if f"Skill List_{skill}" in skill_cols]

        if selected_skills:
            df_skills = df[selected_skills].fillna(0).values
            user_skills = np.ones((1, len(selected_skills)))  # assumes equal weight

            skill_similarity = cosine_similarity(df_skills, user_skills).flatten()
        else:
            skill_similarity = np.zeros(len(df))
    else:
        skill_similarity = np.zeros(len(df))

    # Encode education level numerically
    education_order = [
        "Less than High School", "High School", "Some College",
        "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"
    ]
    user_edu = education_order.index(user_profile["education_level"]) if user_profile["education_level"] in education_order else 0
    job_edu = df["Education Level"].apply(lambda x: education_order.index(x) if x in education_order else 0)
    education_match = 1 - abs(job_edu - user_edu) / len(education_order)

    # Combine similarities (weights can be adjusted)
    final_score = 0.5 * riasec_similarity + 0.3 * skill_similarity + 0.2 * education_match

    df["Score"] = final_score

    # Sort and format results
    top_matches = df.sort_values("Score", ascending=False).head(top_n)

    return top_matches[[
        "Title", "Description", "Score", "Education Level",
        "R", "I", "A", "S", "E", "C"
    ]], {
        "total_jobs_considered": len(df),
        "skills_matched": user_profile["skills"],
        "personalized_message": f"Hello {user_profile.get('user_name', 'there')}, here are your top {top_n} job matches based on your RIASEC, skills, and education!"
    }

# --- Example usage (for testing) ---
if __name__ == "__main__":
    sample_profile = {
        "user_name": "Allan",
        "R": 0.3, "I": 0.6, "A": 0.2, "S": 0.5, "E": 0.1, "C": 0.4,
        "education_level": "Bachelor's Degree",
        "skills": ["Critical Thinking", "Reading Comprehension", "Time Management"]
    }

    recommendations, info = hybrid_similarity_recommender(sample_profile)
    print(info["personalized_message"])
    print(recommendations.head(5))
