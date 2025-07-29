# get_user_profile.py

def get_user_profile(education_level_text, experience_level_text, selected_skills, riasec_scores):
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

    return {
        "normalized_education_score": education_level_map.get(education_level_text, 0),
        "education_label": education_level_text,
        "experience_level": experience_level_map.get(experience_level_text, 0),
        "selected_skills": selected_skills,
        "riasec_scores": riasec_scores
    }

def build_user_profile(form_data):
    """
    Build and return a user profile dictionary based on form input.
    """
    return {
        'user_name': form_data.get('user_name', 'User'),
        'R': form_data.get('R', 0),
        'I': form_data.get('I', 0),
        'A': form_data.get('A', 0),
        'S': form_data.get('S', 0),
        'E': form_data.get('E', 0),
        'C': form_data.get('C', 0),
        'education_level': form_data.get('education_level', 0),
        'skills': form_data.get('skills', []),
    }

def transform_user_profile(user_profile_dict):
    """
    Transform a raw user profile dictionary into a format suitable for similarity comparison.
    """
    import numpy as np

    # Convert the profile into a vector for similarity comparison
    vector = np.array([
        user_profile_dict['R'],
        user_profile_dict['I'],
        user_profile_dict['A'],
        user_profile_dict['S'],
        user_profile_dict['E'],
        user_profile_dict['C'],
        user_profile_dict.get('education_level', 0)
    ])

    return vector.reshape(1, -1)

