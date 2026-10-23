"""Discard every batch a ZeroMQ gate forwards (run on the coordinator for exp6).

exp6 measures the gate alone, but a ZeroMQ PUSH socket with no peer queues and
then blocks, so something has to drain the gate's output. This binds a PULL
socket, counts rows and batches, and prints the totals on Ctrl-C or SIGTERM.

    python lncs/code/experiments/zmq_sink.py --bind tcp://*:5556
"""

import argparse
import os
import signal
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bind", default="tcp://*:5556")
    args = ap.parse_args()

    import zmq
    from onnx_edge_ids import flow_frames

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(args.bind)
    print(f"[OK] ZeroMQ sink on {args.bind}", flush=True)
    rows = batches = 0
    try:
        while True:
            batch = flow_frames.unpack(sock.recv_multipart(copy=False))
            rows += len(batch["x"])
            batches += 1
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print(f"[done] {rows:,} rows in {batches:,} batches", flush=True)
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
