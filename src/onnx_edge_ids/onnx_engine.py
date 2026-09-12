#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from .inference_engine import InferenceEngine


class OnnxInferenceEngine(InferenceEngine):
    def __init__(self, model_path, providers=None, intra_threads=0):
        super().__init__()
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.label_name = None
        self.prob_name = None
        self.intra_threads = intra_threads
        self._load_model(providers)

    def _load_model(self, providers):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("ONNX inference needs onnxruntime installed.") from e

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        opts = ort.SessionOptions()
        if self.intra_threads > 0:
            opts.intra_op_num_threads = self.intra_threads

        requested = providers or ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = [p for p in requested if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.model_path), opts, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self.label_name, self.prob_name = outputs[0], outputs[1]

    def _infer(self, matrix):
        x = np.ascontiguousarray(matrix, dtype=np.float32)
        labels, probs = self.session.run(
            [self.label_name, self.prob_name], {self.input_name: x}
        )
        preds = np.asarray(labels).astype(np.int64).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(len(preds), -1)
        return preds, probs
