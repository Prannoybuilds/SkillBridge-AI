<div align="center">

# 📄🔗💼 Resume ↔ Job Description Match Scorer

### A Siamese BiLSTM Deep Neural Network for Text-Pair Matching

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Siamese%20BiLSTM-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Introduction

Given a **resume** and a **job description**, this project trains a neural network to read both and predict whether the resume is a good match for the role — outputting a single probability between 0 (no match) and 1 (strong match). It was built as a **Day 19 Deep Neural Networks assignment**, and demonstrates a **Siamese** architecture: one shared **Bidirectional LSTM** text encoder used twice (once per document) to compare two pieces of text rather than classify one.

## 🎯 Problem Statement

Comparing two documents for compatibility (resume vs. JD) is fundamentally different from classifying a single document. A plain classifier can't naturally express "how similar are these two texts to each other" — it needs an architecture that encodes both inputs into the *same* vector space so they can be directly compared. This project builds and trains exactly that.

## 🎯 Objectives

- Generate a synthetic but structurally realistic resume/JD dataset across 5 job domains.
- Build a **Siamese BiLSTM** encoder shared between the resume and JD inputs.
- Combine the two encoded vectors (concatenation + absolute difference + element-wise product) into an explicit similarity signal for the classification head.
- Train, evaluate, and save a working match-probability model, with full metrics and visualizations.

## ✨ Features

| Feature | Description |
|---|---|
| 🔗 **Siamese architecture** | One shared BiLSTM encoder used for both the resume and the JD — halves the parameter count vs. two separate encoders and forces both texts into the same embedding space |
| ↔️ **Bidirectional LSTM** | Reads each text forwards *and* backwards so meaning-changing context (e.g. "experience **required**" vs. "experience **preferred**") is captured |
| ➕ **Explicit similarity features** | `\|resume_vector - jd_vector\|` and element-wise product are concatenated in before the final Dense head, giving the model an easy "distance" signal |
| 🧹 **Custom text cleaning** | Preserves technical tokens (`c++`, `ci/cd`, `node.js`) instead of blindly stripping punctuation |
| 🧪 **Synthetic, reproducible dataset** | 900 labeled resume/JD pairs generated across 5 domains — no scraping, no licensing/privacy concerns |
| ⏹️ **Early stopping** | Training halts and restores the best weights once validation loss stops improving |
| 📊 **Full evaluation suite** | Classification report, confusion matrix, and training curves are all saved automatically |
| 🔮 **Standalone inference script** | Score any new (resume, JD) pair with `predict.py` without retraining |

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras (`TextVectorization`, `Embedding`, `Bidirectional LSTM`, custom `AbsDifference` layer) |
| Data handling | `pandas`, `numpy` |
| Evaluation | `scikit-learn` (`classification_report`, `confusion_matrix`, `train_test_split`) |
| Visualization | `matplotlib` |
| Language | Python 3.10+ |

## 🏗️ Project Architecture

```
raw resume text   ─┐
                    ├─> clean_text() ─> TextVectorization ─> Embedding ─> BiLSTM x2 ─┐
raw JD text       ─┘  (shared encoder, same weights used for both)                  ├─> |diff|, product, concat ─> Dense ─> Dense ─> sigmoid ─> match probability
```

**Why this design?**
- **Siamese network** = one shared encoder used twice (once for the resume, once for the JD). This forces both texts into the same vector space so the network can literally measure how "close" they are, while halving the parameters versus training two separate encoders.
- **BiLSTM** inside the encoder reads each sentence forwards *and* backwards, so a word's representation is shaped by the whole sentence, not just the words before it.
- **`|diff|` and elementwise product** give the Dense head an explicit, easy-to-learn "how similar are these two vectors" signal on top of the raw concatenated vectors.

## 📁 Folder Structure

```
resume-jd-match-siamese-bilstm/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── train.jsonl                # synthetic resume/JD dataset (900 labeled pairs)
├── src/
│   ├── make_dataset.py            # generates the synthetic dataset
│   ├── preprocessing.py           # text cleaning + dataframe helpers
│   ├── model.py                   # Siamese BiLSTM architecture
│   ├── train.py                   # trains the model end-to-end
│   ├── evaluate.py                # reloads saved model, reprints metrics
│   └── predict.py                 # score a single new (resume, JD) pair
├── models/
│   ├── resume_jd_match_model.keras
│   └── vectorizer_vocab.json
├── outputs/
│   ├── training_history.png       # loss/accuracy curves
│   ├── training_history.json
│   ├── confusion_matrix.png
│   └── evaluation_report.md
├── assets/                        # extra screenshots (see Screenshots section below)
└── notebooks/
    └── Resume_JD_Match_Prannoy.ipynb   # full standalone walkthrough notebook
```

## 📦 Requirements

```
tensorflow>=2.16
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

## ⚙️ Setup Guide

```bash
git clone https://github.com/Prannoybuilds/resume-jd-match-siamese-bilstm.git
cd resume-jd-match-siamese-bilstm

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ How to Run

```bash
cd src
python make_dataset.py     # regenerate data/train.jsonl (optional, already included)
python train.py            # train the model, writes to models/ and outputs/
python evaluate.py         # reload saved model, print metrics again
python predict.py          # score one example resume/JD pair
```

Alternatively, open `notebooks/Resume_JD_Match_Prannoy.ipynb` for a full, self-contained walkthrough (uses smaller hyperparameters — `VOCAB_SIZE=4000`, `MAX_LEN=48`, `EMBEDDING_DIM=32`, `LSTM_UNITS=24` — than the production `src/train.py` script, which uses `VOCAB_SIZE=8000`, `MAX_LEN=96`, `EMBEDDING_DIM=64`, `LSTM_UNITS=48`).

## 🔄 Workflow

1. **`make_dataset.py`** generates 900 synthetic (resume, JD, label) triples across 5 domains — data science, frontend, backend, devops, marketing. Same-domain pairs are labeled `match` (1); cross-domain pairs are labeled `no_match` (0).
2. **`preprocessing.py`** lowercases and cleans text while preserving technical tokens, then loads the data into a DataFrame.
3. **`train.py`** splits the data 80/20 (stratified), adapts a `TextVectorization` layer on the training text, builds the Siamese BiLSTM model, and trains it with early stopping (`monitor="val_loss", patience=5`).
4. Training curves, a classification report, and a confusion matrix are saved to `outputs/`.
5. The trained model and vectorizer vocabulary are saved to `models/`.
6. **`predict.py`** reloads the saved model to score any new (resume, JD) pair.

## 📸 Screenshots

> The training curves and confusion matrix already exist in `outputs/` — no need to recapture those, just reference them directly (see Results section above). Capture the two extra ones below from your notebook/terminal and drop them into `assets/`.

| # | Screenshot | What it shows |
|---|---|---|
| 1 | ![Model summary](assets/01-model-summary.png) | `model.summary()` output — the Siamese BiLSTM architecture, layer-by-layer |
| 2 | ![Prediction example](assets/02-prediction-example.png) | Terminal output of `predict.py` scoring a real (resume, JD) pair |

<details>
<summary>📋 Exact steps to capture these (click to expand)</summary>

1. **Screenshot 1** — open `notebooks/Resume_JD_Match_Prannoy.ipynb`, find the cell that builds and prints the model (`model.summary()`), and screenshot its output.
2. **Screenshot 2** — in a terminal, run `python src/predict.py` and screenshot the printed match score/probability.
3. Save both as PNGs in `assets/`, named to match the table above.

</details>

## 📊 Results (this run)

| Metric | Value |
|---|---:|
| Validation Accuracy | **0.9778** |
| Validation Loss | 0.0681 |
| Precision | 0.9574 |
| Recall | 1.0000 |

```
              precision    recall  f1-score   support

    no_match       1.00      0.96      0.98        90
       match       0.96      1.00      0.98        90

    accuracy                           0.98       180
   macro avg       0.98      0.98      0.98       180
weighted avg       0.98      0.98      0.98       180
```

The model reached **~97.8% validation accuracy** on the held-out synthetic set (180 pairs), catching every true match (100% recall) with only 4 false-positive "match" predictions out of 90 true no-match pairs. Training and validation loss both decreased smoothly across 8 epochs with no sign of overfitting (see `outputs/training_history.png`), and the confusion matrix (`outputs/confusion_matrix.png`) confirms strong performance in both directions.

## 🐛 Known Issue (Fixed)

The originally submitted `predict.py` was missing an import of the custom `AbsDifference` layer from `model.py`. Without it, Keras can't deserialize the saved `.keras` model (it raises `Could not locate class 'AbsDifference'`) because the class's `@register_keras_serializable` decorator never runs. **Fixed** by adding `from model import AbsDifference` at the top of `predict.py` — `evaluate.py` already had this import correctly. No model, training, or architecture logic was changed.

## 📁 Dataset Note

`data/train.jsonl` is a **synthetically generated** dataset (see `src/make_dataset.py`), built by pairing resumes and job descriptions across 5 domains (data science, frontend, backend, devops, marketing). Same-domain pairs are labeled `match`, cross-domain pairs are labeled `no_match`. This keeps the project fully self-contained and reproducible without depending on a scraped or shared dataset — though it also means the model has learned a somewhat easier, template-shaped version of the real-world matching problem, and results on real resumes/JDs would likely be lower than the 97.8% seen here.

## 🎓 Learning Outcomes

- Why sequence models (RNN/LSTM) are needed for text instead of bag-of-words.
- The vanishing gradient problem in vanilla RNNs, and how LSTM gates address it.
- What "Bidirectional" adds over a plain LSTM.
- Why the encoder is *shared* (Siamese) between two inputs instead of using two separate encoders.
- Why `|diff|` and elementwise product are useful as extra features before the final Dense head.
- Binary cross-entropy loss and what precision/recall/accuracy mean for a match/no-match task.
- Early stopping and why it helps prevent overfitting.

## 🚀 Future Improvements

- Replace the synthetic template-based dataset with real (anonymized) resume/JD pairs for a more realistic difficulty level.
- Add attention or a Transformer-based encoder (e.g. a small pretrained sentence embedding model) as a stronger alternative to the BiLSTM.
- Add a lightweight Streamlit/Gradio demo UI for `predict.py` so non-technical users can try it interactively.
- Track experiments (hyperparameters, metrics) with a tool like MLflow or Weights & Biases.
- Add unit tests for `preprocessing.clean_text()` and the custom `AbsDifference` layer.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Prannoy Sen**
B.Tech Computer Science & Engineering
Summer Training Programme on Machine Learning & Agentic AI — Electronics & ICT Academy, IIT Roorkee

## 🙏 Acknowledgements

- [Electronics & ICT Academy, IIT Roorkee](https://www.eiacindia.org/) — Summer Training Programme on Machine Learning & Agentic AI
- [TensorFlow / Keras](https://www.tensorflow.org/) documentation on `TextVectorization` and custom layers
- [scikit-learn](https://scikit-learn.org/) for evaluation utilities
