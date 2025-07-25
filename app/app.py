# --- Required Libraries ---
from engine import hybrid_similarity_recommender

import streamlit as st
import numpy as np
import pandas as pd
import re
import json
import random
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time
import uuid
from feedback import save_feedback, get_average_rating, load_all_feedback
try:
    from analytics import log_usage, load_usage_data
except ImportError:
    from analytics_stub import log_usage, load_usage_data


# --- Page Configuration ---
st.set_page_config(
    page_title="SmartPath Career Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"] { font-size: 17px !important; }
        .stSlider > div > div { font-size: 17px !important; }
        label, .stSelectbox label, .stMultiSelect label {
            font-size: 17px !important;
            font-weight: 500 !important;
        }
        div.stButton > button:first-child {
            background-color: #0e76a8;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.75em 1.5em;
            font-size: 1.1em;
            border: none;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #095e88;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class="header-container" style='text-align:center; padding: 1rem; background-color: #003262; border-radius: 10px;'>
        <h1 style='color:white;'>🔍 SmartPath Career Recommender</h1>
        <p style='color:white;'>Discover careers aligned with your strengths, passions, and education.<br>
        <em>Powered by the RIASEC Science models and real-world job market data.</em></p>
    </div>
""", unsafe_allow_html=True)

# --- Language Toggle ---
lang = st.radio("🌐 Select Language", ("English", "Kiswahili"), horizontal=True)

# --- Session State Setup ---
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""
if 'name_submitted' not in st.session_state:
    st.session_state['name_submitted'] = False
if 'career_submitted' not in st.session_state:
    st.session_state['career_submitted'] = False

# --- Name Form ---
if not st.session_state['name_submitted']:
    with st.form("user_profile_form"):
        st.subheader("👤 Your Name")
        user_name = st.text_input("Enter your name").strip().title()
        continue_clicked = st.form_submit_button("Continue")

    if continue_clicked:
        if not user_name:
            st.warning("⚠️ Please enter your name.")
        elif not re.match("^[A-Za-z ]+$", user_name):
            st.warning("⚠️ Name must contain only letters and spaces.")
        else:
            st.session_state['user_name'] = user_name
            st.session_state['name_submitted'] = True
            st.rerun()

# --- Career Form ---
if st.session_state['name_submitted']:
    user_name = st.session_state['user_name']

    st.markdown(f"""
        <h3 style="color:#2c7be5;">
            Hi <strong style="color:#28a745;">{user_name}</strong>, Welcome to SmartPath! 🎉
            Please fill in your profile below.
        </h3>
    """, unsafe_allow_html=True)

    with st.form("career_profile_form"):
        st.subheader("🧠 Enter Your RIASEC Scores (0–7)")
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.slider("Realistic (R)", 0.0, 7.0, 4.0, step=0.5)
            i = st.slider("Investigative (I)", 0.0, 7.0, 4.0, step=0.5)
        with col2:
            a = st.slider("Artistic (A)", 0.0, 7.0, 4.0, step=0.5)
            s = st.slider("Social (S)", 0.0, 7.0, 4.0, step=0.5)
        with col3:
            e = st.slider("Enterprising (E)", 0.0, 7.0, 4.0, step=0.5)
            c = st.slider("Conventional (C)", 0.0, 7.0, 4.0, step=0.5)

        st.subheader("🎓 Highest Education Level")
        edu_level = st.selectbox("Select your highest level of education", [
            "Less than High School", "High School Diploma or Equivalent", "Some College Courses",
            "Associate Degree", "Bachelor's Degree", "Master's Degree",
            "Doctoral or Professional Degree", "Post-Doctoral Training"
        ], index=4)

        st.subheader("🛠️ Strong Skills (Select up to 5)")

        try:
            from engine import get_encoded_skill_columns
            skills_list = get_encoded_skill_columns()
        except Exception as e:
            st.error(f"❌ Failed to load skill options: {e}")
            skills_list = []
            
        # Show the multiselect only if skills were loaded
        if skills_list:
            selected_skills = st.multiselect(
                "Select your top skills",
                skills_list,
                max_selections=5
            )
        else:
            st.warning("⚠️ No skills found to display. Please check the Skills.xlsx file.")
            selected_skills = []

        # Validation
        if len(selected_skills) == 0:
            st.warning("⚠️ You haven't selected any skills. This may affect your recommendations.")
            user_skills = []
        else:
            user_skills = selected_skills

        submitted = st.form_submit_button("🚀 Get Career Recommendations")

        if submitted:
            st.session_state['career_submitted'] = True
            st.session_state['form_data'] = {
                'user_name': user_name,
                'R': r, 'I': i, 'A': a, 'S': s, 'E': e, 'C': c,
                'education_level': edu_level,
                'skills': selected_skills
            }

# --- Output Section ---
session_id = str(uuid.uuid4())
if st.session_state.get('career_submitted'):
    st.info("⏳ Generating recommendations...")
    progress = st.progress(0)
    for pct in range(0, 101, 10):
        progress.progress(pct)
        time.sleep(0.05)

    user_profile = st.session_state['form_data']
    try:
        results = hybrid_similarity_recommender(user_profile)
        if isinstance(results, tuple): results = results[0]

        icons = {"Manager": "👔", "Developer": "💻", "Analyst": "📊", "Engineer": "🛠️", "Teacher": "📚", 
                 "Designer": "🎨", "Scientist": "🔬", "Doctor": "🩺", "Nurse": "👩‍⚕️", "Technician": "🔧", 
                 "Consultant": "🧠"}
        def get_icon(title):
            for keyword, icon in icons.items():
                if keyword.lower() in title.lower():
                    return icon
            return "💼"

        if isinstance(results, pd.DataFrame):
            if not results.empty and 'Title' in results.columns and 'Description' in results.columns:
                if "Final Score" in results.columns:
                    results = results.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
                results['Icon'] = results['Title'].apply(get_icon)
                top_job = results.iloc[0]
            else:
                st.error("⚠️ Results missing required columns.")
                st.stop()
        else:
            st.error("⚠️ Unexpected data format. Please try again or contact support.")
            st.stop()

    except Exception as e:
        st.error("⚠️ Unexpected error occurred during recommendation generation.")
        st.exception(e)
    else:
        if results.empty:
            st.warning("No matching jobs found. Try adjusting your input.")
        else:
            st.success(f"🎯 Hi {user_name}, here are your top careers matches based on your interests, skills, and education level.")

            st.markdown(f"""
                <div style="background-color:#fff3cd;padding:15px;border-radius:10px;">
                    <h2 style="color:#00796b;">🌟 Your Top Career Match: {top_job['Icon']} <span style="color:#d32f2f;">{top_job['Title']}</span></h2>
                    <p style="font-size:16px;">{top_job['Description'][:250]}...</p> 
                </div> 
            """, unsafe_allow_html=True)

            st.markdown("### 📌 Top Career Matches")
            visible_results = results.drop(columns=['R', 'I', 'A', 'S', 'E', 'C'])
            st.dataframe(visible_results, use_container_width=True)

            # Download button
        with st.container():
            csv = visible_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Recommendations",
                data=csv,
                file_name="career_recommendations.csv",
                mime='text/csv'
            )

            # Sort by total score to ensure meaningful bar order
            top5 = results.sort_values(by='Hybrid Recommendation Score', ascending=False).head(5).copy()

            # Melt the DataFrame for Altair
            melted = pd.melt(
                top5,
                id_vars=["Title"],
                value_vars=[
                    "User RIASEC Similarity", 
                    "Normalized Education Score", 
                    "User Skill Similarity"
                ],
                var_name="Metric", value_name="Score"
            ) 
             
            # Hybrid Recommendation Chart
            st.markdown("### 🧮 Final Hybrid Recommendation Score (Top 10)")

            bar_chart = alt.Chart(results.head(10)).mark_bar().encode(
                x=alt.X("Hybrid Recommendation Score:Q", title="Score"),
                y=alt.Y("Title:N", sort='-x'),
                tooltip=["Title", "Hybrid Recommendation Score"]
            ).properties(
                width=700,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)

            # --- Optional Filters ---
            st.markdown("### 🎛️ Filter Score Breakdown")

            metric_filter = st.selectbox(
                "Select which metric to display",
                ["All", "User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"]
            )

            if metric_filter != "All":
                filtered_data = melted[melted["Metric"] == metric_filter]
            else:
                filtered_data = melted

            # --- Score Breakdown ---
            st.markdown("### 📊 Score Breakdown for Top 5 Jobs")
            top5 = results.head(5).copy()
            
            # Altair chart with fixed colors
            chart = alt.Chart(filtered_data).mark_bar().encode(
                x=alt.X("Score:Q", stack="zero"),
                y=alt.Y("Title:N", sort='-x'),
                color=alt.Color("Metric:N",
                    scale=alt.Scale(
                        domain=["User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"],
                        range=["#1f77b4", "#2ca02c", "#d62728"]  # Blue, Green, Red
                    )
                ),
                tooltip=["Title", "Metric", "Score"]
            ).properties(width="container", height=400)

            st.altair_chart(chart, use_container_width=True)

            # --- Auto-Generated Interpretation ---
            st.markdown("### 📘 Interpretation")

            score_means = top5[[
                "User RIASEC Similarity", 
                "Normalized Education Score", 
                "User Skill Similarity"
            ]].mean().round(2)

            best_metric = score_means.idxmax()
            st.info(f"Your strongest matching factor across the top jobs is **{best_metric}** (average score: {score_means[best_metric]})")

            weakest_metric = score_means.idxmin()
            st.warning(f"The lowest average contributor is **{weakest_metric}**. Consider improving this area to unlock more opportunities.")

            # --- RIASEC Radar Chart ---
            st.markdown("### 🧭 Your RIASEC Personality Fit vs. Top Career")
            riasec_labels = ['R', 'I', 'A', 'S', 'E', 'C']
            user_values = [user_profile[code] for code in riasec_labels]
            top_job_values = [top_job[code] for code in riasec_labels]
            user_values += user_values[:1]
            top_job_values += top_job_values[:1]
            riasec_labels += riasec_labels[:1]

            euclidean_distance = np.linalg.norm(np.array(user_values[:-1]) - np.array(top_job_values[:-1]))
            match_score = int(100 - (euclidean_distance / (np.sqrt(len(user_values[:-1])) * 7)) * 100)
            match_score = max(0, min(match_score, 100))

            log_usage(
                session_id=session_id,
                user_name=user_name,
                riasec_scores={"R": r, "I": i, "A": a, "S": s, "E": e, "C": c},
                education=edu_level,
                skills=selected_skills,
                top_match=top_job['Title'],
                match_score=match_score
            )

            st.markdown(f"🎯 **Match Score: {match_score}%** – Alignment with *{top_job['Title']}* role")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=user_values, theta=riasec_labels, fill='toself', name='You',
                                          line=dict(color='royalblue')))
            fig.add_trace(go.Scatterpolar(r=top_job_values, theta=riasec_labels, fill='toself',
                                          name=top_job['Title'], line=dict(color='darkorange')))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 7])), showlegend=True,
                              width=700, height=550)
            st.plotly_chart(fig, use_container_width=True)

            alignment_traits = [code for code in riasec_labels[:-1] if abs(user_profile[code] - top_job[code]) <= 1.0]
            misaligned_traits = [code for code in riasec_labels[:-1] if abs(user_profile[code] - top_job[code]) > 1.0]
            st.markdown("#### 🔎 Alignment Insight")
            st.success(f"You closely align with this role in: **{', '.join(alignment_traits)}**")
            if misaligned_traits:
                st.info(f"Consider developing traits related to: **{', '.join(misaligned_traits)}** for a stronger fit.")

            # --- ✨ Job Description Display ---
            st.markdown("### 📝 Detailed Overview of Your Top Career")
            with st.expander(f"🔍 {top_job['Icon']} **{top_job['Title']}**", expanded=True):
                st.markdown(f"""
                    <div style="line-height: 1.7;">
                        <p>{top_job['Description']}</p>
                        <hr style="margin: 10px 0;">
                        <p><strong>Suggested Skills:</strong> <em>{', '.join(user_profile['skills'])}</em></p>
                        <p><strong>Education Required:</strong> <em>{user_profile['education_level']}</em></p>
                    </div>
                """, unsafe_allow_html=True)

            # --- Optional Fun Career ---
            if st.checkbox("🎉 Surprise Career Match (just for fun!)"):
                random_job = results[results["Title"] != top_job["Title"]].sample(1).iloc[0]
                st.info(f"💡 **{random_job['Icon']} {random_job['Title']}**\n\n{random_job['Description'][:250]}...")

# --- 📣 Feedback Section ---
st.markdown("---")
st.subheader("💬 We'd love your feedback!")

with st.form("feedback_form"):
    col1, col2 = st.columns([1, 3])

    with col1:
        rating = st.slider("How would you rate your results?", 1, 5, 3)
    with col2:
        comment = st.text_area("Any comments or suggestions?")

    submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        anon = st.checkbox("Submit anonymously", value=False)
        save_feedback(rating, comment, session_id, user_name if not anon else None)
        st.success("✅ Thank you! Your feedback has been recorded.")

        avg_rating = get_average_rating()
        if avg_rating:
            st.markdown(f"⭐ Average user rating so far: **{avg_rating}/5**")

# Emoji Feedback
st.markdown("Or leave a quick emoji reaction to your result:")

emoji_col1, emoji_col2, emoji_col3 = st.columns(3)

with emoji_col1:
    if st.button("😊 Yes, I liked it"):
        save_feedback(5, "Positive emoji reaction", session_id)
        st.toast("Thanks for the smile!")
with emoji_col2:
    if st.button("😐"):
        save_feedback(3, "Neutral emoji reaction", session_id)
        st.toast("Thanks for your input!")
with emoji_col3:
    if st.button("😞"):
        save_feedback(1, "Negative emoji reaction", session_id)
        st.toast("Sorry to hear that. We'll improve!")

# section for admin or developers
with st.expander("📊 View All Feedback (Admin Only)"):
    df_feedback = load_all_feedback()

# --- Insights Dashboard ---
with st.expander("📈 User Insights Dashboard"):
    usage_df = load_usage_data()

    if not usage_df.empty:
        st.metric("Total Users", usage_df["session_id"].nunique())

        st.markdown("### 🎓 Education Levels")
        st.bar_chart(usage_df["education"].value_counts())

        st.markdown("### 💼 Most Recommended Careers")
        st.bar_chart(usage_df["top_match"].value_counts())

        st.markdown("### 🧠 Avg Match Score by RIASEC")
        for trait in ["R", "I", "A", "S", "E", "C"]:
            avg_scores = usage_df.groupby(trait)["match_score"].mean().reset_index()
            st.line_chart(avg_scores.set_index(trait))
    else:
        st.info("No user usage data yet.")

# --- Footer ---
st.markdown("""  
<hr style="margin-top: 50px; margin-bottom: 10px;">
<div style='
    text-align: center;
    font-size: 0.9rem;
    color: gray;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ccc;'>
    &copy; 2025 <strong>SmartPath</strong> &mdash; Developed by <strong>Allan Ofula</strong> <br>
    Youth Advocate & Data Scientist | Moringa School Capstone Project
</div>
""", unsafe_allow_html=True)

