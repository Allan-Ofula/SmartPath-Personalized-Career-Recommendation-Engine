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
            st.experimental_rerun()

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

        st.subheader("🛠️ Strong Skills (Select up to 10)")
        skill_options = [
            "Data Analysis", "Communication", "Problem Solving", "Project Management", "Creativity",
            "Critical Thinking", "Leadership", "Teamwork", "Technical Writing", "Machine Learning",
            "SQL", "Python", "R", "Tableau", "Excel", "Public Speaking", "Negotiation", "Sales",
            "Graphic Design", "Customer Service", "Financial Literacy", "Coding", "UX/UI Design",
            "Time Management", "Oral Comprehension", "Written Comprehension", "Originality",
            "Deductive Reasoning", "Inductive Reasoning", "Flexibility of Closure", "Visualization",
            "Reaction Time", "Speech Clarity"
        ]
        selected_skills = st.multiselect("Select your top skills", skill_options, max_selections=10)

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
if st.session_state.get('career_submitted'):
    st.info("⏳ Generating recommendations...")
    progress = st.progress(0)

    for pct in range(0, 101, 10):
        progress.progress(pct)
        time.sleep(0.05)

    user_profile = st.session_state['form_data']

    try:
        results = hybrid_similarity_recommender(user_profile)
        if isinstance(results, tuple):
            results = results[0]

        icons = {
            "Manager": "👔", "Developer": "💻", "Analyst": "📊", "Engineer": "🛠️",
            "Teacher": "📚", "Designer": "🎨", "Scientist": "🔬", "Doctor": "🩺",
            "Nurse": "👩‍⚕️", "Technician": "🔧", "Consultant": "🧠"
        }

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
                    <h2 style="color:#00796b;">🌟 Your Top Career Match: <span style="color:#d32f2f;">{top_job['Title']}</span></h2>
                    <p style="font-size:16px;">{top_job['Description'][:250]}...</p> 
                </div> 
            """, unsafe_allow_html=True)

            st.markdown("### 📌 Top Career Matches")
            st.dataframe(results.drop(columns=['R', 'I', 'A', 'S', 'E', 'C']), use_container_width=True)

            # Score Breakdown Chart
            st.markdown("### 📊 Score Breakdown for Top 5 Jobs")
            top5 = results.head(5).copy()
            melted = pd.melt(
                top5,
                id_vars=["Title"],
                value_vars=["User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"],
                var_name="Metric",
                value_name="Score"
            )
            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("Score:Q", stack="zero"),
                y=alt.Y("Title:N", sort='-x'),
                color="Metric:N",
                tooltip=["Title", "Metric", "Score"]
            ).properties(width="container", height=400)
            st.altair_chart(chart, use_container_width=True)

            # Radar Chart
            st.markdown("### 🧭 RIASEC Match (Top Career vs You)") 
            riasec_labels = ['R', 'I', 'A', 'S', 'E', 'C']
            user_values = [user_profile[code] for code in riasec_labels]
            top_job_values = [top_job[code] for code in riasec_labels]
            user_values += user_values[:1]
            top_job_values += top_job_values[:1]
            riasec_labels += riasec_labels[:1]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=user_values, theta=riasec_labels, fill='toself', name='You', line=dict(color='blue')))
            fig.add_trace(go.Scatterpolar(r=top_job_values, theta=riasec_labels, fill='toself', name=top_job['Title'], line=dict(color='orange')))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 7])), showlegend=True, width=600, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Job Description
            st.markdown("### 📝 Description of Top Job")
            st.info(f"**{top_job['Title']}**\n\n{top_job['Description']}")

            if st.checkbox("🎉 Surprise Career Match (just for fun!)"):
                random_job = results.sample(1).iloc[0]
                st.info(f"💡 **{random_job['Icon']} {random_job['Title']}**\n\n{random_job['Description'][:250]}...")
