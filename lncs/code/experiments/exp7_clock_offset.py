"""Clock offset of a board against this host, over one SSH session.

Sends N requests through a single SSH channel; the board answers each with its
time.time(). offset = board - midpoint(send, receive) for the request with the
smallest round trip, whose half-RTT bounds the error. Run on the coordinator:

    python lncs/code/experiments/exp7_clock_offset.py bvdung@192.168.1.81 > j1_offset.json
"""

import json
import subprocess
import sys
import time

REMOTE = ("python3 -u -c 'import sys,time\n"
          "for _ in sys.stdin: print(repr(time.time()), flush=True)'")


def main():
    host, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, REMOTE],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    samples = []
    for _ in range(n):
        t0 = time.time()
        p.stdin.write("x\n")
        p.stdin.flush()
        remote = float(p.stdout.readline())
        t1 = time.time()
        samples.append((t1 - t0, remote - (t0 + t1) / 2))
        time.sleep(0.05)
    p.stdin.close()
    p.wait()
    rtt, offset = min(samples)
    rtts = sorted(s[0] for s in samples)
    print(json.dumps({
        "host": host,
        "taken_at": time.time(),
        "samples": n,
        "offset_s": round(offset, 6),
        "error_bound_s": round(rtt / 2, 6),
        "rtt_min_ms": round(rtt * 1000, 2),
        "rtt_median_ms": round(rtts[len(rtts) // 2] * 1000, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
