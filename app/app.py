# --- Required Libraries ---
import streamlit as st
from recommender_engine import generate_recommendations
import numpy as np
import pandas as pd
import altair as alt
import os
import re
import json
import random
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

FEEDBACK_FILE = "feedback.csv"

def save_feedback(rating, comment, session_id, user_name=None):
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_name": user_name if user_name else "Anonymous",
        "rating": rating,
        "comment": comment
    }

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_entry])

    df.to_csv(FEEDBACK_FILE, index=False)

def get_average_rating():
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        if not df.empty:
            return round(df["rating"].mean(), 2)
    return None
session_id = str(uuid.uuid4())

# --- Page Config ---
st.set_page_config(
    page_title="SmartPath Career Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Metadata ---
@st.cache_data
def load_metadata():
    df = pd.read_csv("data/job_profiles_clean.csv")
    skill_cols = [c for c in df.columns if c.startswith("Skill List_")]
    skill_options = [c.replace("Skill List_", "") for c in skill_cols]
    max_edu_norm = df["Normalized Education Score"].max()
    return skill_cols, skill_options, max_edu_norm

skill_cols, skill_options, max_edu_norm = load_metadata()

# --- Session State Setup ---
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""
if 'name_submitted' not in st.session_state:
    st.session_state['name_submitted'] = False
if 'career_submitted' not in st.session_state:
    st.session_state['career_submitted'] = False

# --- Global Styling ---
st.markdown("""
    <style>
        html, body {
            font-size: 17px !important;
            background-color: #f0f4f8 !important;
        }
        .stSlider > div > div { font-size: 17px !important; }
        label, .stSelectbox label, .stMultiSelect label {
            font-size: 17px !important; font-weight: 500 !important;
        }
        div.stButton > button:first-child {
            background-color: #0e76a8; color: white; font-weight: bold;
            border-radius: 8px; padding: 0.75em 1.5em; font-size: 1.1em;
            border: none; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #095e88; transform: scale(1.03);
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

# --- Name Input ---
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

# --- Profile Form ---
if st.session_state['name_submitted']:
    user_name = st.session_state['user_name']
    st.markdown(f"""
        <h3 style="color:#2c7be5;">
            Hi <strong style="color:#28a745;">{user_name}</strong>, Welcome to SmartPath! 🎉
            Please fill in your profile below.
        </h3>
    """, unsafe_allow_html=True)

    with st.form("Career_profile_form"):
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
        ui_edu_levels = [
            "Less than High School", "High School Diploma or Equivalent", "Some College Courses",
            "Associate Degree", "Bachelor's Degree", "Master's Degree",
            "Doctoral or Professional Degree", "Post-Doctoral Training"
        ]
        edu_level = st.selectbox("Select your highest level of education", ui_edu_levels, index=0)

        st.subheader("🛠️ Strong Skills (Select up to 5)")
        selected_skills_ui = st.multiselect("Choose your strongest skills", skill_options, max_selections=5)

        submitted = st.form_submit_button("🚀 Get Career Recommendations")
        if submitted:
            st.session_state["submitted"] = True

# --- Output Logic ---
if st.session_state["submitted"]:
    st.info("Analyzing your profile and generating matches...")
    st.markdown("<hr style='border: 4px solid #003262; border-radius: 10px;'>", unsafe_allow_html=True)

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
    edu_num = edu_mapping[edu_level]
    user_edu_norm = edu_num / max(edu_mapping.values())
    user_skills = ["Skill List_" + s for s in selected_skills_ui]

    user_profile = {
        'R': r, 'I': i, 'A': a, 'S': s, 'E': e, 'C': c,
        'education_level': user_edu_norm,
        'skills': user_skills
    }

    st.info("Recommendations generated successfully.")
    st.info(f"Hi \n{user_name}, here are your top careers matches based on your interests, skills, and education level.")

    try:
        results, _ = generate_recommendations(user_profile)
    except Exception as err:
        st.error("Something went wrong. Please try again.")
        st.exception(err)
    else:
        if results.empty:
            st.warning("No matching careers found for this combination.")
        else:
            sort_metric = st.selectbox("🔽 Sort Top Careers By", [
                "Hybrid Recommendation Score",
                "User RIASEC Similarity", 
                "Education Similarity", 
                "User Skill Similarity"
            ])
            results = results.sort_values(sort_metric, ascending=False)
            top5 = results.head(10).reset_index(drop=True)
            top5.index = top5.index + 1

            highlight = top5.iloc[0]

            st.markdown("""
            ### 🌟 Your Top Career Match:
            <div style='background-color: #fff3cd; border: 2px solid #dc3545; padding: 1em; border-radius: 12px;'>
                <h3 style='color: green;'><strong>💼 {}</strong></h3>
                <p style='font-size: 16px; color: #333;'>{}</p>
            </div>
            """.format(highlight['Title'], highlight['Description']), unsafe_allow_html=True)

            st.markdown("### 📌 Top Career Matches")
            st.dataframe(top5, use_container_width=True)

            st.info("""
            #### 📘 Interpretation:
            - **Higher values** indicate stronger alignment.
            - **User RIASEC Similarity**: Match with interests
            - **User Skill Similarity**: Match with skills
            - **Education Score**: Education fit with job
            """)

            st.markdown("### 📊 Hybrid Recommendation Score Breakdown (Top 5 Careers)")

            melted = results.head(5).melt(
                id_vars=["Title"],
                value_vars=["User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"],
                var_name="Metric",
                value_name="Score"
            )

            color_map = {
                "User RIASEC Similarity": "#1f77b4",
                "Normalized Education Score": "#2ca02c",
                "User Skill Similarity": "#ff7f0e"
            }

            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Title:N", title="Job Title", sort='-x'),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                    legend=alt.Legend(
                        orient="bottom",  # change to "bottom" if you want it below
                        title="Metric Breakdown"
                    )
                ),
                tooltip=["Title", "Metric", "Score"]
            ).properties(
                width="container",
                height=400
            )

            st.altair_chart(chart, use_container_width=True)

            st.markdown("### 📈 Average Scores Across Top 5")
            avg_scores = results.head(5)[["User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"]].mean()
            st.write(avg_scores.to_frame("Average Score"))

            st.markdown("### 🔸 RIASEC Radar Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[r, i, a, s, e, c], theta=['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'], fill='toself'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 7])), width=700, height=550, showlegend=True)
            st.plotly_chart(fig)

            st.markdown("""
                > _This radar chart visualizes how your personality aligns across the six RIASEC dimensions. Peaks indicate stronger traits. Your top career matches are more aligned with your dominant RIASEC traits._

            **RIASEC Meanings:**
            - R: Practical, hands-on
            - I: Analytical, science-driven
            - A: Creative, expressive
            - S: Social, helper
            - E: Leader, business
            - C: Organized, structured

            **Alignment Insight:**
            You closely align with: <strong>{}</strong>
            """.format(max(zip(['R','I','A','S','E','C'], [r,i,a,s,e,c]), key=lambda x: x[1])[0]), unsafe_allow_html=True)

            st.markdown("### 📝 Detailed Overview of Your Top Careers")
            for _, row in results.head().iterrows():
                with st.expander(f"🔹 {row['Title']}"):
                    st.write(row['Description'])
                    st.write(f"**Education Level:** {row['Education Level']} — _{row['Education Category Label']}_")
                    st.write(f"**Preparation Level:** {row['Preparation Level']}")
                    st.write(f"**RIASEC Scores:** R={row['R']}, I={row['I']}, A={row['A']}, S={row['S']}, E={row['E']}, C={row['C']}")

            # --- Optional Fun Career ---
            st.markdown("### 🎉 Surprise Career Match (just for fun!)")
            fun_career = results.sample(1).iloc[0]
            st.info(f"💼 **{fun_career['Title']}** — {fun_career['Description']}")
            
            # --- Insights Dashboard ---
            with st.expander("📈 User Insights Dashboard"):
                if not results.empty:
                    st.markdown("### 🎓 Education Levels")
                    edu_avg_scores = results.groupby("Education Category Label")["Hybrid Recommendation Score"].mean()
                    st.bar_chart(edu_avg_scores)

                    st.markdown("### 💼 Most Recommended Careers")
                    top_titles = results['Title'].value_counts().head(10)
                    st.bar_chart(top_titles)

                    st.markdown("### 🧠 Avg Match Score by RIASEC")
                    avg_scores = results[["R", "I", "A", "S", "E", "C"]].mean()
                    st.line_chart(avg_scores)

                else:
                    st.info("No results data available for insights.")

            with st.expander("🔐 Admin Section"):
                admin_pw = st.text_input("Enter Admin Password", type="password")
                if admin_pw == "admin123":
                    st.subheader("Total Users")
                    st.write("(Placeholder) Total Users Count: 128")
            # --- Optional Access Control ---
            with st.expander("🔐 Admin Panel"):
                is_admin = st.checkbox("I am an admin")

                if is_admin and os.path.exists(FEEDBACK_FILE):
                    feedback_df = pd.read_csv(FEEDBACK_FILE)
                    st.dataframe(feedback_df)

                    csv = feedback_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download feedback.csv",
                        data=csv,
                        file_name="feedback.csv",
                        mime="text/csv"
                    )


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
    Youth Advocate | Data Scientist 
</div>
""", unsafe_allow_html=True)
