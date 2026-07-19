"""
train.py
---------
End-to-end training script:
  1. Load + clean data
  2. Train/validation split
  3. Build text vectorizer + Siamese BiLSTM model
  4. Train with early stopping
  5. Save model, tokenizer vocab, training history, and evaluation report
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from preprocessing import load_jsonl, add_clean_columns
from model import build_text_vectorizer, build_siamese_match_model

VOCAB_SIZE = 8000
MAX_LEN = 96
EMBEDDING_DIM = 64
LSTM_UNITS = 48
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42


def make_tf_dataset(df, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "resume_input": df["resume_clean"].values,
            "jd_input": df["jd_clean"].values,
        },
        df["label"].values.astype("float32"),
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs("../outputs", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    print("Loading data...")
    df = load_jsonl("../data/train.jsonl")
    df = add_clean_columns(df)
    print(f"Total examples: {len(df)}")
    print(df["label"].value_counts())

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    print("Adapting text vectorizer...")
    combined_text = pd.concat([train_df["resume_clean"], train_df["jd_clean"]])
    vectorizer = build_text_vectorizer(VOCAB_SIZE, MAX_LEN, combined_text.values)

    train_ds = make_tf_dataset(train_df, shuffle=True)
    val_ds = make_tf_dataset(val_df, shuffle=False)

    print("Building model...")
    model = build_siamese_match_model(vectorizer, VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---- Save training history plot ----
    hist_df = pd.DataFrame(history.history)
    hist_df.to_json("../outputs/training_history.json", orient="records", indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["loss"], label="train_loss")
    plt.plot(hist_df["val_loss"], label="val_loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_df["accuracy"], label="train_acc")
    plt.plot(hist_df["val_accuracy"], label="val_acc")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/training_history.png", dpi=120)
    plt.close()

    # ---- Evaluation ----
    print("Evaluating...")
    eval_dict = model.evaluate(val_ds, return_dict=True)
    val_probs = model.predict(val_ds).ravel()
    val_preds = (val_probs >= 0.5).astype(int)
    y_true = val_df["label"].values

    report = classification_report(y_true, val_preds, target_names=["no_match", "match"])
    print(report)

    with open("../outputs/evaluation_report.md", "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Metrics\n\n")
        for k, v in eval_dict.items():
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write("\n## Classification Report\n\n```\n")
        f.write(report)
        f.write("\n```\n")

    cm = confusion_matrix(y_true, val_preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["pred_no_match", "pred_match"])
    plt.yticks([0, 1], ["actual_no_match", "actual_match"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("../outputs/confusion_matrix.png", dpi=120)
    plt.close()

    # ---- Save model ----
    model.save("../models/resume_jd_match_model.keras")
    print("Saved model to ../models/resume_jd_match_model.keras")

    # Save vectorizer vocabulary so predict.py can rebuild the same vectorizer
    vocab = vectorizer.get_vocabulary()
    with open("../models/vectorizer_vocab.json", "w") as f:
        json.dump(vocab, f)
    print("Saved vectorizer vocabulary.")


if __name__ == "__main__":
    main()
