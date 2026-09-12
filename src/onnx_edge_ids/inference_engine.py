#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatchPrediction:
    predictions: np.ndarray
    confidences: np.ndarray
    stats: dict


class InferenceEngine:
    LABEL_MAP = {0: "Benign", 1: "Attack"}

    def __init__(self):
        self.total_predictions = 0
        self.total_attacks = 0
        self.total_inference_time = 0.0

    def _infer(self, matrix):
        raise NotImplementedError

    def predict_batch(self, matrix):
        start = time.perf_counter()
        preds, probs = self._infer(matrix)
        inference_time = (time.perf_counter() - start) * 1000.0

        count = int(preds.shape[0])
        attacks = int(np.count_nonzero(preds == 1))
        confidences = (
            probs[np.arange(count), preds].astype(float)
            if count and probs is not None else np.zeros(count)
        )

        self.total_predictions += count
        self.total_attacks += attacks
        self.total_inference_time += inference_time

        return BatchPrediction(
            predictions=preds,
            confidences=confidences,
            stats={
                "batch_size": count,
                "attacks_found": attacks,
                "inference_time_ms": round(inference_time, 3),
                "avg_time_ms": round(inference_time / count, 3) if count else 0,
            },
        )

    def predict_single(self, matrix):
        result = self.predict_batch(matrix[:1])
        prediction = int(result.predictions[0])
        return {
            "prediction": prediction,
            "label": self.LABEL_MAP.get(prediction, "Unknown"),
            "confidence": float(result.confidences[0]),
            "inference_time_ms": result.stats["inference_time_ms"],
            "is_attack": prediction == 1,
        }

    def get_stats(self):
        avg_time = (
            self.total_inference_time / self.total_predictions
            if self.total_predictions else 0.0
        )
        return {
            "total_predictions": self.total_predictions,
            "total_attacks": self.total_attacks,
            "attack_rate": (
                self.total_attacks / self.total_predictions
                if self.total_predictions else 0.0
            ),
            "avg_inference_time_ms": round(avg_time, 3),
            "total_inference_time_ms": round(self.total_inference_time, 3),
        }

    def close(self):
        pass


def create_inference_engine(engine, model_path, providers=None):
    name = engine.strip().lower()
    if name == "onnx":
        from .onnx_engine import OnnxInferenceEngine
        return OnnxInferenceEngine(model_path=model_path, providers=providers)
    if name == "numpy":
        from .numpy_engine import NumpyInferenceEngine
        return NumpyInferenceEngine(model_path=model_path)
    raise ValueError(f"Unsupported engine '{engine}'. Use 'onnx' or 'numpy'.")
