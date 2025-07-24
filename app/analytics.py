import pandas as pd
import os
from datetime import datetime

USAGE_LOG = "usage_data.csv"

def log_usage(session_id, user_name, riasec_scores, education, skills, top_match, match_score):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "user_name": user_name,
        **riasec_scores,  # dictionary with R, I, A, S, E, C
        "education": education,
        "skills": ", ".join(skills),
        "top_match": top_match,
        "match_score": match_score,
    }

    df = pd.DataFrame([data])
    if os.path.exists(USAGE_LOG):
        df.to_csv(USAGE_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(USAGE_LOG, mode='w', header=True, index=False)

def load_usage_data():
    if os.path.exists(USAGE_LOG):
        return pd.read_csv(USAGE_LOG)
    return pd.DataFrame()
