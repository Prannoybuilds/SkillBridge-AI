"""
evaluate.py
------------
Reloads the saved model + dataset and reports validation metrics.
Useful for demoing the project without re-running training.
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_jsonl, add_clean_columns
from model import AbsDifference  # noqa: F401 (needed so keras can deserialize custom layer)

SEED = 42
BATCH_SIZE = 64


def make_tf_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "resume_input": df["resume_clean"].values,
            "jd_input": df["jd_clean"].values,
        },
        df["label"].values.astype("float32"),
    ))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    df = load_jsonl("../data/train.jsonl")
    df = add_clean_columns(df)
    _, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])

    model = tf.keras.models.load_model("../models/resume_jd_match_model.keras")
    val_ds = make_tf_dataset(val_df)

    eval_dict = model.evaluate(val_ds, return_dict=True)
    print("Metrics:", eval_dict)

    probs = model.predict(val_ds).ravel()
    preds = (probs >= 0.5).astype(int)
    y_true = val_df["label"].values

    print(classification_report(y_true, preds, target_names=["no_match", "match"]))
    print("Confusion matrix:\n", confusion_matrix(y_true, preds))


if __name__ == "__main__":
    main()
