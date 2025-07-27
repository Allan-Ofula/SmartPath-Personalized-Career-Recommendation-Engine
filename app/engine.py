def hybrid_similarity_recommender(user_profile, top_n=10):
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from pathlib import Path
    from get_user_profile import get_user_profile

    # --- Load Dataset ---
    data_path = Path(__file__).resolve().parent / "data" / "job_profiles_clean.csv"
    df = pd.read_csv(data_path)

    # --- Education & Experience Mapping ---
    education_level_map = {
        "Less than High School": 0,
        "High School Diploma or Equivalent": 4,
        "Some College Courses": 5,
        "Associate Degree": 6,
        "Bachelor's Degree": 8,
        "Master's Degree": 10,
        "Doctoral or Professional Degree": 12,
        "Post-Doctoral Training": 14
    }

    experience_level_map = {
        "None or short demonstration": 0,
        "Up to and including 1 month": 1,
        "Anything beyond short demonstration, up to and including 1 month": 2,
        "Over 1 month, up to and including 3 months": 3,
        "Over 3 months, up to and including 6 months": 4,
        "Over 6 months, up to and including 1 year": 5,
        "Over 1 year, up to and including 2 years": 6,
        "Over 2 years, up to and including 4 years": 7,
        "Over 4 years, up to and including 6 years": 8,
        "Over 6 years, up to and including 8 years": 9,
        "Over 8 years, up to and including 10 years": 10,
        "Over 10 years": 11
    }

    # --- Convert Education Column ---
    df['Education Numeric'] = df['Education Level'].map(education_level_map).fillna(0)
    df['Normalized Education Score'] = df['Education Numeric'] / max(education_level_map.values())

    # --- Filter jobs that require education <= user level ---
    user_edu_numeric = education_level_map.get(user_profile.get('education_level', ''), 0)

    # Filter by user education level - only keep jobs requiring equal or lower education level
    df = df[df['Education Numeric'] <= user_edu_numeric]

    if df.empty:
        return pd.DataFrame(), {"num_recommendations": 0}

    # --- Normalize Experience ---
    df['Experience Numeric'] = df['Preparation Level'].map(experience_level_map).fillna(0)
    df['Normalized Experience Score'] = df['Experience Numeric'] / max(experience_level_map.values())

    # --- RIASEC Similarity ---
    user_riasec = np.array([user_profile['riasec_scores'].get(trait, 0) for trait in ['R', 'I', 'A', 'S', 'E', 'C']])
    user_riasec_normalized = user_riasec / np.sum(user_riasec)
    career_riasec = df[['R', 'I', 'A', 'S', 'E', 'C']].values
    df['User RIASEC Similarity'] = cosine_similarity([user_riasec_normalized], career_riasec)[0]

    # --- Skill Match ---
    selected_skills = user_profile.get('selected_skills', [])
    if selected_skills:
        for skill in selected_skills:
            col = f"Skill List_{skill}"
            if col not in df.columns:
                df[col] = 0
        skill_cols = [f"Skill List_{skill}" for skill in selected_skills]
        df['User Skill Similarity'] = df[skill_cols].sum(axis=1) / len(skill_cols)
    else:
        df['User Skill Similarity'] = 0

    # --- Final Hybrid Score ---
    WEIGHTS = {
        "riasec": 0.45,
        "skills": 0.2,
        "education": 0.35,
        "experience": 0.0
    }

    df['Hybrid Recommendation Score'] = (
        WEIGHTS["riasec"] * df['User RIASEC Similarity'] +
        WEIGHTS["skills"] * df['User Skill Similarity'] +
        WEIGHTS["education"] * df['Normalized Education Score'] +
        WEIGHTS["experience"] * df['Normalized Experience Score']
    )

    top_matches = df.sort_values(by='Hybrid Recommendation Score', ascending=False).head(top_n)

    return (
        top_matches[[
            'Title', 'Description', 'Education Level', 'Preparation Level', 'Education Category Label',
            'User RIASEC Similarity', 'User Skill Similarity',
            'Normalized Education Score', 'Normalized Experience Score',
            'Hybrid Recommendation Score', 'R', 'I', 'A', 'S', 'E', 'C'
        ]],
        {"num_recommendations": len(top_matches)}
    )