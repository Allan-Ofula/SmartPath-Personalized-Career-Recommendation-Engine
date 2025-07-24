import pandas as pd
from datetime import datetime
import os

FEEDBACK_FILE = "data/feedback.csv"

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# Make Feedback Anonymous or User-Linked
def save_feedback(rating, comment, session_id, user_name=None):
    """Save feedback entry with optional user_name."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": timestamp,
        "session_id": session_id,
        "rating": rating,
        "comment": comment,
        "user_name": user_name if user_name else "Anonymous"
    }

    # Append to file
    df = pd.DataFrame([data])
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["timestamp", "session_id", "rating", "comment", "user_name"])

def get_average_rating():
    df = load_feedback()
    if not df.empty:
        return round(df["rating"].mean(), 2)
    return None

# Show Feedback Results in a Chart (Optional Admin View)
def load_all_feedback():
    """
    Load all feedback from the CSV file as a DataFrame.
    Returns:
        df: pandas DataFrame of all feedback entries
    """
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        return df
    return pd.DataFrame(columns=["timestamp", "session_id", "rating", "comment"])
