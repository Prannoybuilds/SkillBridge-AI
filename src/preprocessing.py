"""
preprocessing.py
-----------------
Text cleaning + dataframe utilities for the Resume <-> JD matching project.

Design choice: we keep technical tokens intact (c++, node.js, ci/cd,
scikit-learn) because stripping punctuation blindly would turn meaningful
skill tokens into garbage. This mirrors real-world resume text where
special characters carry signal.
"""

import json
import re
import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    """Load a .jsonl file of {resume, job_description, label} rows into a DataFrame."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    """Lowercase, collapse whitespace, and keep tech-friendly characters."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    # keep letters, digits, + # . - / and spaces (covers c++, c#, ci/cd, node.js)
    text = re.sub(r"[^a-z0-9+#./\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["resume_clean"] = df["resume"].apply(clean_text)
    df["jd_clean"] = df["job_description"].apply(clean_text)
    df["resume_len"] = df["resume_clean"].apply(lambda x: len(x.split()))
    df["jd_len"] = df["jd_clean"].apply(lambda x: len(x.split()))
    return df
