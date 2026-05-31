import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextProcessor:
    MAX_LEN = 100

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        tokenizer_path = os.path.join(base_dir, "ecg_model_hierarchical", "tokenizer_v6.pkl")
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

    def process(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            text = "no report"
        sequence = self.tokenizer.texts_to_sequences([text])
        padded   = pad_sequences(sequence, maxlen=self.MAX_LEN, padding="post")
        return np.array(padded, dtype=np.float32)