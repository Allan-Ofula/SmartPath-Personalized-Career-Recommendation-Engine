def hybrid_similarity_recommender(user_profile, top_n=10):
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from pathlib import Path

    # --- Education Level Mapping ---
    education_level_map = {
        # Experience-like levels (mapped to 0)
        "None or short demonstration": 0,
        "Up to and including 1 month": 0,
        "Anything beyond short demonstration, up to and including 1 month": 0,
        "Over 1 month, up to and including 3 months": 0,
        "Over 3 months, up to and including 6 months": 0,
        "Over 6 months, up to and including 1 year": 0,
        "Over 1 year, up to and including 2 years": 0,
        "Over 2 years, up to and including 4 years": 0,
        "Over 4 years, up to and including 6 years": 0,
        "Over 6 years, up to and including 8 years": 0,
        "Over 8 years, up to and including 10 years": 0,
        "Over 10 years": 0,

        # Actual education levels
        "High School Diploma or the equivalent": 1,
        "Some College Courses": 2,
        "Post-Secondary Certificate": 3,
        "Associate's Degree (or other 2-year degree)": 4,
        "Bachelor's Degree": 5,
        "Post-Baccalaureate Certificate": 6,
        "Master's Degree": 7,
        "Post-Master's Certificate": 8,
        "First Professional Degree": 9,
        "Doctoral Degree": 10,
        "Post-Doctoral Training": 11
    }

    # --- Experience Level Mapping ---
    experience_level_mapping = {
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

    # --- Load Dataset ---
    data_path = Path(__file__).resolve().parent / "data" / "job_profiles_clean.csv"
    df = pd.read_csv(data_path)

    # --- Normalize Education ---
    df['Education Numeric'] = df['Education Level'].map(education_level_map).fillna(0)
    df['Normalized Education Score'] = df['Education Numeric'] / max(education_level_map.values())

    # --- Normalize Experience ---
    df['Experience Numeric'] = df['Preparation Level'].map(experience_level_mapping).fillna(0)
    df['Normalized Experience Score'] = df['Experience Numeric'] / max(experience_level_mapping.values())

    # --- RIASEC Similarity ---
    user_riasec = np.array([user_profile[trait] for trait in ['R', 'I', 'A', 'S', 'E', 'C']])
    user_riasec_normalized = user_riasec / np.sum(user_riasec)
    career_riasec = df[['R', 'I', 'A', 'S', 'E', 'C']].values
    df['User RIASEC Similarity'] = cosine_similarity([user_riasec_normalized], career_riasec)[0]

    # --- Skill Match ---
    selected_skills = user_profile.get('skills', [])
    if selected_skills:
        for skill in selected_skills:
            skill_col = f"Skill List_{skill}"
            if skill_col not in df.columns:
                df[skill_col] = 0
        skill_cols = [f"Skill List_{skill}" for skill in selected_skills]
        df['User Skill Similarity'] = df[skill_cols].sum(axis=1) / len(selected_skills)
    else:
        df['User Skill Similarity'] = 0

    # --- Hybrid Score Calculation ---
    df['Hybrid Recommendation Score'] = (
        0.45 * df['User RIASEC Similarity'] +
        0.25 * df['User Skill Similarity'] +
        0.15 * df['Normalized Education Score'] +
        0.15 * df['Normalized Experience Score']
    )

    # --- Top N Matches ---
    top_matches = df.sort_values(by='Hybrid Recommendation Score', ascending=False).head(top_n)

    return (
        top_matches[[
            'Title', 'Description', 'Education Level', 'Preparation Level',
            'User RIASEC Similarity', 'User Skill Similarity',
            'Normalized Education Score', 'Normalized Experience Score',
            'Hybrid Recommendation Score', 'R', 'I', 'A', 'S', 'E', 'C'
        ]],
        {
            "num_recommendations": len(top_matches)
        }
    )
