# get_user_profile.py
def get_user_profile(education_level, experience_level, selected_skills, riasec_scores):
    """
    Prepares the user profile dictionary based on input fields.
    """
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

    return {
        "education_level": education_level_map.get(education_level, 0),
        "experience_level": experience_level_map.get(experience_level, 0),
        "selected_skills": selected_skills,
        "riasec_scores": riasec_scores
    }

def build_user_profile(form_data):
    return {
        'user_name': form_data.get('user_name', ''),
        'riasec_scores': {
            'R': form_data.get('R', 0),
            'I': form_data.get('I', 0),
            'A': form_data.get('A', 0),
            'S': form_data.get('S', 0),
            'E': form_data.get('E', 0),
            'C': form_data.get('C', 0),
        },
        'education_level': form_data.get('education_level', ''),
        'skills': form_data.get('skills', [])  # 👈 change key here
    }