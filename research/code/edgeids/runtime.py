"""Running a served graph: ONNX Runtime labels and probabilities, NumPy traversals, and label comparisons."""

import numpy as np

from edgeids.forest import predict_lockstep
from edgeids.onnx_forest import raw_scores


def ort_outputs(path, x, batch=50000):
    """(label output, label derived from the probability output) of ONNX Runtime.

    The derived label is class 1 iff P(class 1) > 0.5, read from column 1 of the
    probability output. For a binary TreeEnsembleClassifier with a single weight
    column, ONNX Runtime's own label output changed meaning between releases
    (1.19: P > 0.5; 1.27: P > 0), so the derived label is what the parity tests
    compare against, and the disagreement between the two is reported separately.
    """
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    labs, probs = [], []
    for i in range(0, len(x), batch):
        out = s.run(None, {name: x[i:i + batch]})
        labs.append(np.asarray(out[0]).reshape(-1))
        probs.append(np.asarray(out[1]))
    lab = np.concatenate(labs).astype(np.int64)
    prob = np.concatenate(probs)
    derived = (prob[:, 1] > 0.5).astype(np.int64) if prob.shape[1] == 2 else np.argmax(prob, axis=1).astype(np.int64)
    return lab, derived


def ort_labels(path, x, batch=50000):
    return ort_outputs(path, x, batch)[1]


def flips(a, b):
    return int((np.asarray(a).astype(np.int64) != np.asarray(b).astype(np.int64)).sum())


def probabilities(forest, scores):
    """The probability output ONNX Runtime returns for these scores (binary models)."""
    s = scores
    if forest.n_targets == 1:
        p1 = 1.0 / (1.0 + np.exp(-s[:, 0])) if forest.post_transform == "LOGISTIC" else s[:, 0]
        return np.stack([1.0 - p1, p1], axis=1)
    if forest.post_transform == "SOFTMAX":
        z = np.exp(s - s.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
    if forest.post_transform == "LOGISTIC":
        return 1.0 / (1.0 + np.exp(-s))
    return s


def numpy_probs(forest, x, dtype, batch=50000):
    return np.concatenate([probabilities(forest, raw_scores(forest, x[i:i + batch], dtype))
                           for i in range(0, len(x), batch)])


def ort_probs(path, x, batch=50000):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([np.asarray(s.run(None, {name: x[i:i + batch]})[1], dtype=np.float64)
                           for i in range(0, len(x), batch)])


def label(p):
    # class 1 iff P(class 1) > 0.5, as exp5 derives ONNX Runtime's label
    return (p[:, 1] > 0.5).astype(np.int64)


def session_labels(session, x, batch=20000):
    name = session.get_inputs()[0].name
    return np.concatenate([
        np.asarray(session.run(None, {name: x[i:i + batch]})[0]).reshape(-1)
        for i in range(0, len(x), batch)
    ]).astype(np.int64)


def lockstep_labels(forest, x, batch=2000):
    return np.concatenate([predict_lockstep(forest, x[i:i + batch])[0] for i in range(0, len(x), batch)])
