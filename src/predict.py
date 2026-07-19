"""
predict.py
-----------
Loads the trained Siamese BiLSTM model and scores a new (resume, job_description) pair.

Usage:
    python predict.py
(edit SAMPLE_RESUME / SAMPLE_JD below, or import predict_match() elsewhere)
"""

import tensorflow as tf
from preprocessing import clean_text
from model import AbsDifference  # noqa: F401 (needed so keras can deserialize custom layer)

MODEL_PATH = "../models/resume_jd_match_model.keras"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_match(model, resume_text: str, jd_text: str) -> dict:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    inputs = {
        "resume_input": tf.constant([resume_clean]),
        "jd_input": tf.constant([jd_clean]),
    }
    prob = float(model.predict(inputs, verbose=0)[0][0])
    label = "MATCH" if prob >= 0.5 else "NO MATCH"
    return {"match_probability": round(prob, 4), "prediction": label}


SAMPLE_RESUME = (
    "Candidate is a 3-5 years experience professional. Core skills include "
    "python, pandas, scikit-learn, tensorflow, sql, machine learning. "
    "Previously worked on projects involving data science systems. "
    "Looking for a role as Data Scientist."
)

SAMPLE_JD = (
    "We are hiring a Machine Learning Engineer with 3-5 years experience. "
    "Required skills: python, deep learning, statistics, nlp, tensorflow. "
    "The candidate will work on data science systems in a collaborative team environment."
)

if __name__ == "__main__":
    model = load_model()
    result = predict_match(model, SAMPLE_RESUME, SAMPLE_JD)
    print("Resume:", SAMPLE_RESUME[:80], "...")
    print("JD:", SAMPLE_JD[:80], "...")
    print("Result:", result)
