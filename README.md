# Capstone Project- SmartPath: Personalized Career Recommendation Engine
Empowering informed career decisions through intelligent, data-driven recommendations.

## Overview
SmartPath is a personalized career recommendation system that leverages the **O\*NET occupational database** to align job seekers—especially students and youth in underserved communities—with the careers best suited to their **skills**, **interests (RIASEC)**, and **education level**.

Unlike generic career counseling tools, SmartPath uses a **hybrid similarity model** (RIASEC, skills, and education) to deliver **personalized** and **actionable** career suggestions, helping users identify their best-fit occupations and the skills they need to thrive.

---

## Key Features

Collects user input for:
- RIASEC interest scores (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
- Educational attainment
- Self-identified strong skills

Computes:
- Cosine similarity between user interests and occupational profiles
- Skill match scores based on selected user strengths
- Education gap analysis

Outputs:
- Top career matches with hybrid similarity scores
- Matched skills and gaps
- CSV export + optional email delivery
- Summary visualization of top 3 job recommendations

---

## Data Sources
SmartPath leverages curated occupational data from reliable public resources to ensure accuracy and relevance in career recommendations.

| Dataset                           | Description                                                                                                                                            | Source                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| **O\*NET 27.0 Database**          | A comprehensive dataset with detailed job descriptions, required skills, education levels, and interest profiles (RIASEC) for hundreds of occupations. | [O\*NET Official Site](https://www.onetcenter.org/database.html)                              |
| **O\*NET Interests Profile**      | Contains standardized RIASEC scores for each occupation, used for matching user interests.                                                             | [O\*NET Work Styles & Interests](https://www.onetonline.org/find/descriptor/result/4.A.1.a.1) |
| **O\*NET Skills Importance**      | Lists the relative importance of key skills (e.g., Critical Thinking, Systems Analysis) across different roles.                                        | [O\*NET Skills Summary](https://www.onetonline.org/skills/)                                   |
| **O\*NET Education Requirements** | Details typical education levels associated with each occupation. Used for computing education similarity scores.                                      | [O\*NET Education Data](https://www.onetonline.org/find/descriptor/browse/Education/)         |

The datasets were cleaned, transformed, and combined into a unified job profile format (job_profiles_clean.csv) used for real-time matching.

Cleaned Dataset Sample (hosted for testing/demo):./data/job_profiles_clean.csv

---

## Tech Stack

- **Python 3.10+**
- **Pandas**, **NumPy**, **Scikit-learn**
- **O\*NET Dataset** (curated version)
- **Matplotlib / Seaborn** (optional for visualizations)
- **Jupyter Notebook** (Interactive UI)
- **Email Integration**: via `smtplib` and `EmailMessage`

---

## How It Works

1. **User Input**  
   The user provides RIASEC scores, top 3 skills (optional), and their highest education level via a simple interface.

2. **Profile Matching**  
   The system compares the user profile to thousands of job profiles from O\*NET using cosine similarity and skill/education matching.

3. **Recommendation Generation**  
   Jobs are scored and ranked based on:
   - RIASEC similarity (40%)
   - Skill match (30%)
   - Education alignment (30%)

4. **Output Delivery**  
   The top 10 jobs are displayed and saved in a CSV file. Users can also opt to receive the file via email.

---

## Future Improvements
- Web-based interface using Flask or Streamlit
- Integration with LinkedIn skills or resume parsing
- Career pathway suggestions (entry → mid-level → senior roles)
- Visual dashboard of occupational trends per country

---

## Acknowledgments
We would like to express our sincere gratitude to:

- Moringa School – for providing the learning foundation and project framework.
- O*NET (Occupational Information Network) – for the rich job dataset that powers this recommendation engine.
- Career Development Theorists – especially John Holland, for the RIASEC model.
- Our instructors Mildred Jepkosgei and Brian Chacha (Moringa School) – for mentorship and support in fostering innovative talent.
This work reflects a growing commitment to applying data science in empowering youth, career clarity, and digital transformation in Africa.

---

## Final Note
SmartPath is more than a tool, it's a vision to democratize access to smart, personalized career insights for every student, dreamer, and job seeker across the globe.

--- 

**Rachael Nyawira**  
Kenya | Data Science Learner | Passionate about using data to transform lives  
Email:
[GitHub](https://github.com/yourgithubusername) | [LinkedIn](https://www.linkedin.com/in/yourlinkedin/)

**Beryl Okelo**  
Kenya | Data Science Learner | Passionate about using data to transform lives  
Email: 
[GitHub](https://github.com/yourgithubusername) | [LinkedIn](https://www.linkedin.com/in/yourlinkedin/)

**Beth Nyambura**  
Kenya | Data Science Learner | Passionate about using data to transform lives  
Email:  
[GitHub](https://github.com/yourgithubusername) | [LinkedIn](https://www.linkedin.com/in/yourlinkedin/)

**Allan Ofula**  
Kenya | Data Science Associate | Youth Advocate | Developer of SmartPath | Passionate about using data to transform lives  
[Email:] ofulaallan@gmail.com  
[GitHub](https://github.com/Allan-Ofula) | [LinkedIn](https://www.linkedin.com/in/allan-ofula-b2804911b/)

**Eugene Maina**  
Kenya | Data Science Learner | Passionate about using data to transform lives  
Email: 
[GitHub](https://github.com/yourgithubusername) | [LinkedIn](https://www.linkedin.com/in/yourlinkedin/)

---
