import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# -----------------------------
# LOAD DATA (FOR TOKENIZER)
# -----------------------------
df = pd.read_csv("ecoli_synthetic_training_data_large.csv")

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df["dna_seq"].astype(str).values)

# -----------------------------
# LOAD MODEL (IMPORTANT FIX)
# -----------------------------
model = load_model("trained_model.h5", compile=False)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_score(dna_sequence):
    seq = tokenizer.texts_to_sequences([dna_sequence])
    seq = pad_sequences(seq, maxlen=300)
    prediction = model.predict(seq, verbose=0)
    return float(prediction[0][0])
