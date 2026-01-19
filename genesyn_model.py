import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# -----------------------------
# LOAD DATASET (REAL COLUMNS)
# -----------------------------
df = pd.read_csv("ecoli_synthetic_training_data_large.csv")

sequences = df["dna_seq"].astype(str).values   # ✅ FIXED
scores = df["score"].values

# -----------------------------
# TOKENIZATION (CHAR LEVEL)
# -----------------------------
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)

X = tokenizer.texts_to_sequences(sequences)
X = pad_sequences(X, maxlen=300)
y = scores

# -----------------------------
# BUILD BiLSTM MODEL
# -----------------------------
model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=16,
        input_length=300
    ),
    Bidirectional(LSTM(64)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("trained_model.h5")
print("✅ AI model trained & saved successfully")
