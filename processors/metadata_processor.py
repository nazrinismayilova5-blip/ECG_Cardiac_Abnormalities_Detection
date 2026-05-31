import numpy as np


class MetadataProcessor:
    def __init__(self):
        pass  # No scaler — manual normalization matches training

    def process(self, metadata: dict, ecg: np.ndarray = None, text: str = "") -> np.ndarray:
        age    = float(metadata.get("age", 0))
        sex    = float(metadata.get("sex", 0))
        height = float(metadata.get("height", 0))
        weight = float(metadata.get("weight", 0))

        bmi = weight / ((height / 100) ** 2) if height > 50 else 0
        if bmi > 100 or bmi < 10:
            bmi = 25.0

        height_norm = height / 200
        weight_norm = weight / 150
        ecg_mean    = float(ecg.mean()) if ecg is not None else 0.0
        ecg_std     = float(ecg.std())  if ecg is not None else 0.0
        has_text    = 1.0 if text and text.strip() else 0.0
        report_len  = float(len(text))

        features = np.array([[
            age, sex, bmi,
            height_norm, weight_norm,
            ecg_mean, ecg_std,
            has_text, report_len
        ]], dtype=np.float32)

        return features