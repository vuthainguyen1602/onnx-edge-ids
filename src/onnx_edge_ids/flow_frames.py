"""Multipart wire format for flow batches sent over ZeroMQ.

A batch travels as one small JSON header frame followed by one raw frame per
array: the (rows, cols) float32 feature matrix and, optionally, per-row
columns such as the producer's send time, the gate's forward time and the
anomaly score. The receiver views each frame in place with ``np.frombuffer``
instead of decoding JSON per record, and the per-row timestamps keep the
end-to-end latency measurement of the Kafka path available.

The module only needs NumPy; frames may be ``bytes``, ``memoryview`` or
``zmq.Frame`` objects (anything exposing ``.buffer`` or the buffer protocol).
"""

import json

import numpy as np

VERSION = 1

# Per-row columns that may follow the feature matrix, with their wire dtype.
COLUMNS = {
    "sent_at": np.float64,
    "forwarded_at": np.float64,
    "score": np.float32,
}


def pack(x, **columns):
    """Return the frames for one batch: header, feature matrix, then each column.

    ``x`` must be a 2-D float32 matrix; ``columns`` are per-row arrays named
    in ``COLUMNS``. Arrays already C-contiguous in the wire dtype are sent
    without a copy, so pass the frames to ``send_multipart(..., copy=False)``.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"feature matrix must be 2-D, got shape {x.shape}")
    rows, cols = x.shape
    names = []
    frames = [None, x]
    for name, values in columns.items():
        if values is None:
            continue
        if name not in COLUMNS:
            raise ValueError(f"unknown column '{name}'; expected one of {sorted(COLUMNS)}")
        values = np.ascontiguousarray(values, dtype=COLUMNS[name])
        if values.shape != (rows,):
            raise ValueError(f"column '{name}' has shape {values.shape}, expected ({rows},)")
        names.append(name)
        frames.append(values)
    header = {"v": VERSION, "rows": rows, "cols": cols, "columns": names}
    frames[0] = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return frames


def _buffer(frame):
    return memoryview(frame.buffer if hasattr(frame, "buffer") else frame)


def unpack(frames, cols=None):
    """Return ``{"x": matrix, <column>: values, ...}`` viewing the received frames.

    Raises ``ValueError`` when the header, frame count or frame sizes do not
    match, or when ``cols`` is given and differs from the sender's width.
    The returned arrays are read-only views; copy them before writing.
    """
    if len(frames) < 2:
        raise ValueError(f"expected a header and a feature frame, got {len(frames)} frame(s)")
    header = json.loads(bytes(_buffer(frames[0])).decode("utf-8"))
    if header.get("v") != VERSION:
        raise ValueError(f"unsupported flow-frame version {header.get('v')!r}")
    rows, width = int(header["rows"]), int(header["cols"])
    if cols is not None and width != cols:
        raise ValueError(f"sender has {width} features, receiver expects {cols}")
    names = header.get("columns", [])
    if len(frames) != 2 + len(names):
        raise ValueError(f"header lists {len(names)} column(s) but {len(frames) - 2} followed")

    def view(frame, dtype, shape):
        buf = _buffer(frame)
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if buf.nbytes != expected:
            raise ValueError(f"frame has {buf.nbytes} bytes, expected {expected} for {shape}")
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    out = {"x": view(frames[1], np.float32, (rows, width))}
    for name, frame in zip(names, frames[2:]):
        if name not in COLUMNS:
            raise ValueError(f"unknown column '{name}' in header")
        out[name] = view(frame, COLUMNS[name], (rows,))
    return out


def concat(batches):
    """Merge unpacked batches into one; a column missing from any batch is dropped."""
    if len(batches) == 1:
        return batches[0]
    shared = set(batches[0]).intersection(*batches[1:])
    return {name: np.concatenate([b[name] for b in batches]) for name in shared}
