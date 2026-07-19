"""
model.py
---------
Siamese BiLSTM network for Resume <-> Job Description matching.

Why "Siamese"?
We have TWO text inputs (resume, JD) that need to be compared. Instead of
building two separate encoders, a Siamese network uses ONE shared encoder
for both inputs. This is important because:
  1. It halves the number of parameters we need to train.
  2. It forces both texts into the SAME embedding space, so "5 years of
     Python experience" (resume) and "requires Python experience" (JD)
     end up close together geometrically if they are semantically related.

Why "BiLSTM" inside the encoder?
A plain LSTM reads left-to-right only, so at word i it only knows what came
before it. A Bidirectional LSTM (BiLSTM) reads the sentence forwards AND
backwards and concatenates both hidden states, so every word's
representation is informed by the FULL sentence, not just the words before it.

Architecture:
    resume_input ---\
                      >-- shared_encoder --> resume_vector --\
    jd_input     ---/                                          >-- combine --> Dense --> match probability
                       shared_encoder --> jd_vector          --/
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="resume_jd_matcher")
class AbsDifference(layers.Layer):
    """Element-wise |a - b|. A proper Layer subclass (instead of a raw Lambda)
    so the saved .keras model can be reloaded safely without enabling
    unsafe deserialization."""

    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)


def build_text_vectorizer(vocab_size: int, max_len: int, adapt_texts):
    """Builds and adapts a TextVectorization layer on the training corpus."""
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None,   # we already cleaned text ourselves in preprocessing.py
        split="whitespace",
    )
    vectorizer.adapt(adapt_texts)
    return vectorizer


def build_shared_encoder(vectorizer, vocab_size, embedding_dim, lstm_units, name="shared_text_encoder"):
    """
    One encoder tower, reused for both resume text and JD text.

    Flow: raw string -> token ids -> embeddings -> BiLSTM -> fixed-size vector
    """
    text_in = layers.Input(shape=(1,), dtype=tf.string, name="raw_text")

    x = vectorizer(text_in)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm_1",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units),
        name="bilstm_2",
    )(x)

    x = layers.Dense(embedding_dim, activation="relu", name="text_projection")(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs=text_in, outputs=x, name=name)


def build_siamese_match_model(vectorizer, vocab_size=8000, embedding_dim=64, lstm_units=48):
    """
    Full Siamese BiLSTM model that takes (resume, job_description) and
    outputs a single match probability in [0, 1].
    """
    resume_input = layers.Input(shape=(1,), dtype=tf.string, name="resume_input")
    jd_input = layers.Input(shape=(1,), dtype=tf.string, name="jd_input")

    encoder = build_shared_encoder(vectorizer, vocab_size, embedding_dim, lstm_units)

    resume_vec = encoder(resume_input)
    jd_vec = encoder(jd_input)

    # Feature combination: concatenate the two vectors plus their
    # element-wise absolute difference and product. The difference/product
    # terms explicitly give the Dense head an easy signal for "how similar
    # are these two vectors", which speeds up learning considerably.
    diff = AbsDifference(name="abs_difference")([resume_vec, jd_vec])
    prod = layers.Multiply(name="elementwise_product")([resume_vec, jd_vec])

    merged = layers.Concatenate(name="feature_merge")([resume_vec, jd_vec, diff, prod])

    x = layers.Dense(64, activation="relu", name="head_dense_1")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu", name="head_dense_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="match_probability")(x)

    model = Model(inputs=[resume_input, jd_input], outputs=output, name="siamese_bilstm_resume_jd_matcher")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model
