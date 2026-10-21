"""The ZeroMQ flow-batch frames must round-trip exactly and reject mismatched frames."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import flow_frames  # noqa: E402


def _batch(rows=7, cols=29, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    sent_at = 1.7e9 + rng.random(rows)
    return x, sent_at


def _wire(frames):
    """What a receiver gets: each frame as its own bytes object."""
    return [bytes(memoryview(f)) if not isinstance(f, bytes) else f for f in frames]


def test_round_trip_keeps_values_and_dtypes():
    x, sent_at = _batch()
    score = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    got = flow_frames.unpack(_wire(flow_frames.pack(x, sent_at=sent_at, score=score)), cols=29)
    assert set(got) == {"x", "sent_at", "score"}
    assert got["x"].dtype == np.float32 and np.array_equal(got["x"], x)
    assert got["sent_at"].dtype == np.float64 and np.array_equal(got["sent_at"], sent_at)
    assert np.array_equal(got["score"], score)


def test_pack_sends_contiguous_float32_without_copy():
    x, _ = _batch()
    assert flow_frames.pack(x)[1] is x


def test_unpack_views_the_frame_in_place():
    x, _ = _batch()
    frames = _wire(flow_frames.pack(x))
    got = flow_frames.unpack(frames)
    assert got["x"].base is not None and not got["x"].flags.writeable


def test_none_columns_are_left_out():
    x, _ = _batch()
    got = flow_frames.unpack(_wire(flow_frames.pack(x, sent_at=None)))
    assert set(got) == {"x"}


def test_empty_batch_round_trips():
    x = np.empty((0, 29), dtype=np.float32)
    got = flow_frames.unpack(_wire(flow_frames.pack(x, sent_at=np.empty(0))))
    assert got["x"].shape == (0, 29) and got["sent_at"].shape == (0,)


def test_concat_merges_batches_and_drops_partial_columns():
    x1, s1 = _batch(3, seed=1)
    x2, s2 = _batch(4, seed=2)
    a = flow_frames.unpack(_wire(flow_frames.pack(x1, sent_at=s1, score=np.zeros(3))))
    b = flow_frames.unpack(_wire(flow_frames.pack(x2, sent_at=s2)))
    got = flow_frames.concat([a, b])
    assert set(got) == {"x", "sent_at"}
    assert np.array_equal(got["x"], np.vstack([x1, x2]))
    assert np.array_equal(got["sent_at"], np.concatenate([s1, s2]))


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda f: f[:1], "header and a feature frame"),
        (lambda f: f[:2], "followed"),
        (lambda f: [f[0], f[1][:-4], f[2]], "bytes, expected"),
        (lambda f: [f[0].replace(b'"v":1', b'"v":9'), *f[1:]], "version"),
    ],
)
def test_mismatched_frames_are_rejected(mangle, message):
    x, sent_at = _batch()
    frames = _wire(flow_frames.pack(x, sent_at=sent_at))
    with pytest.raises(ValueError, match=message):
        flow_frames.unpack(mangle(frames))


@pytest.mark.parametrize(
    "header",
    [b"\xff\xfe", b"not json", b"[1, 2]", b'{"v":1}', b'{"v":1,"rows":"x","cols":29}',
     b'{"v":1,"rows":-1,"cols":29}'],
)
def test_malformed_header_raises_value_error(header):
    # the nodes drop a batch on ValueError and would crash on anything else
    x, _ = _batch()
    frames = _wire(flow_frames.pack(x))
    with pytest.raises(ValueError):
        flow_frames.unpack([header, *frames[1:]])


def test_feature_width_is_checked():
    x, _ = _batch(cols=28)
    with pytest.raises(ValueError, match="28 features"):
        flow_frames.unpack(_wire(flow_frames.pack(x)), cols=29)


def test_pack_rejects_bad_columns():
    x, _ = _batch()
    with pytest.raises(ValueError, match="unknown column"):
        flow_frames.pack(x, label=np.zeros(len(x)))
    with pytest.raises(ValueError, match="expected"):
        flow_frames.pack(x, sent_at=np.zeros(len(x) + 1))


def test_zeromq_round_trip():
    zmq = pytest.importorskip("zmq")
    x, sent_at = _batch(100)
    ctx = zmq.Context()
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    try:
        pull.bind("inproc://flow-frames-test")
        push.connect("inproc://flow-frames-test")
        push.send_multipart(flow_frames.pack(x, sent_at=sent_at), copy=False)
        got = flow_frames.unpack(pull.recv_multipart(copy=False), cols=29)
        assert np.array_equal(got["x"], x) and np.array_equal(got["sent_at"], sent_at)
    finally:
        push.close(linger=0)
        pull.close(linger=0)
        ctx.term()
