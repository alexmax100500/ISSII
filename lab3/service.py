from __future__ import annotations

import bentoml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import onnxruntime as ort
import time

EXAMPLE_INPUT = {"X_test": [[0.5, 1.2, -0.3], [0.8, -0.9, 1.4]]}  # Example input

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Summarization:

    def __init__(self) -> None:
        # Load the ONNX model and start the inference session
        self.sess = ort.InferenceSession("logreg_iris.onnx")
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name

        # Load dataset for testing
        self.dataset = pd.read_csv("dataset.csv")

    @bentoml.api
    def predict(self, input_sample: dict = EXAMPLE_INPUT) -> dict:
        """
        Perform inference on the given input sample.
        Preprocess the input, run the ONNX model, and return results.
        """
        # Preprocess input
        X_test = np.array(input_sample["X_test"], dtype=np.float32)
        y_test = np.array(input_sample.get("y_test", []))

        # Measure inference time
        start_time = time.time()
        y_onx = self.sess.run([self.label_name], {self.input_name: X_test})[0]
        end_time = time.time()

        # Post-process results
        inference_time = end_time - start_time
        response = {"predictions": y_onx.tolist(), "inference_time": inference_time}

        # Include accuracy if ground-truth labels are provided
        if len(y_test) > 0:
            accuracy = accuracy_score(y_test, y_onx)
            response["accuracy"] = accuracy

        return response