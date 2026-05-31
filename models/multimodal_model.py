import os
import numpy as np
import tensorflow as tf


class MultimodalModel:
    def __init__(self, binary_path: str, abn_path: str):
        base_dir = os.path.dirname(os.path.dirname(__file__))

        binary_full = os.path.join(base_dir, binary_path)
        abn_full    = os.path.join(base_dir, abn_path)

        self.model_binary = tf.saved_model.load(binary_full)
        self.model_abn    = tf.saved_model.load(abn_full)

        self.infer_binary = self.model_binary.signatures["serving_default"]
        self.infer_abn    = self.model_abn.signatures["serving_default"]

        self._binary_specs = self.infer_binary.structured_input_signature[1]
        self._abn_specs    = self.infer_abn.structured_input_signature[1]

        print("\nBINARY MODEL INPUTS:")
        for k, v in self._binary_specs.items():
            print(k, v.shape)

        print("\nABNORMAL MODEL INPUTS:")
        for k, v in self._abn_specs.items():
            print(k, v.shape)

    def _build_inputs(self, specs, ecg_t, meta_t, text_t):

        inputs = {}

        for name, spec in specs.items():

            shape = spec.shape

            # ECG input
            if len(shape) == 3:
                inputs[name] = ecg_t

            # TEXT input
            elif shape[-1] == 100:
                inputs[name] = text_t

            # METADATA input
            elif shape[-1] == 9:
                inputs[name] = meta_t

        print("INPUTS SENT:", inputs.keys())

        return inputs

    def predict(self, ecg: np.ndarray, meta: np.ndarray, text: np.ndarray):

        ecg_t = tf.constant(ecg, dtype=tf.float32)
        meta_t = tf.constant(meta, dtype=tf.float32)
        text_t = tf.constant(text, dtype=tf.float32)

        binary_inputs = self._build_inputs(
            self._binary_specs,
            ecg_t,
            meta_t,
            text_t
        )

        binary_output = self.infer_binary(**binary_inputs)

        binary_prob = list(binary_output.values())[0].numpy()[0][0]

        print("\nBinary abnormal probability:", binary_prob)

        abn_inputs = self._build_inputs(
            self._abn_specs,
            ecg_t,
            meta_t,
            text_t
        )

        abn_output = self.infer_abn(**abn_inputs)

        abn_probs = list(abn_output.values())[0].numpy()[0]

        print("Abnormal class probabilities:", abn_probs)
        norm_prob = 1.0 - binary_prob

        full_probs = [
            float(norm_prob),
            float(abn_probs[0] * binary_prob),
            float(abn_probs[1] * binary_prob),
            float(abn_probs[2] * binary_prob),
            float(abn_probs[3] * binary_prob),
        ]

        total = sum(full_probs)

        full_probs = [p / total for p in full_probs]

        print("Final probabilities:", full_probs)

        return [full_probs]