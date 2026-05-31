import numpy as np


class ECGProcessor:
    EXPECTED_SHAPE = (1000, 12)

    def __init__(self):
        self.last_processed: np.ndarray | None = None

    def process(self, file) -> np.ndarray:
        if not file.filename.endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        data = np.loadtxt(file.file, delimiter=",")

        if data.shape != self.EXPECTED_SHAPE:
            raise ValueError(
                f"ECG must be shape {self.EXPECTED_SHAPE}, got {data.shape}."
            )

        if np.isnan(data).any():
            raise ValueError("ECG data contains NaN values.")

        # Lead-wise normalization
        data = (data - data.mean(axis=0, keepdims=True)) / \
               (data.std(axis=0, keepdims=True) + 1e-8)

        result = data.reshape(1, 1000, 12)
        self.last_processed = result
        return result