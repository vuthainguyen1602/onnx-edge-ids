"""Can the early-exit undecided rate serve as a label-free drift alarm?

exp4 observed that the share of flows the exact early-exit rule cannot settle at
its first check rises from 4.8% in distribution to 28.2% on shifted traffic (0.7%
to 68.1% on gate-forwarded flows). This experiment turns the observation into a
calibrated test:

1. compute the per-flow "undecided at the first check" flag on both test splits
   with the deployed schedule (first check after 104 trees, then every 8);
2. cut each split into windows of W flows and take the undecided rate per window;
3. calibrate an alarm threshold on the first half of the in-distribution windows
   (their maximum rate, plus the same rule with a mean + 6 sd variant);
4. report the false-alarm rate on the held-out in-distribution windows and the
   detection rate on the shifted windows.

The test splits carry no timestamps, but their row order follows the source
captures' label phases (runs of a few dozen rows of one label), so a window of W
consecutive rows is a traffic segment with its own attack mix (2% to 22% attacks
per 10,000-row window in 2017), not a uniform random draw. That mix variation is
what the in-distribution spread measures and what a threshold has to absorb. The
windows are still not timed traffic, and the calibration covers one shift.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from edgeids.data import ARTIFACTS, RESULTS, Gate, load_test_split
from edgeids.forest import load_forest, margin_bounds, predict_early_exit

BATCH = 2000


def undecided_flags(forest, bounds, schedule, x):
    used = np.concatenate([
        predict_early_exit(forest, x[i:i + BATCH], schedule, bounds=bounds)[1]
        for i in range(0, len(x), BATCH)
    ])
    return used > schedule[0]


def window_rates(flags, w):
    n = len(flags) // w
    if n == 0:
        return np.empty(0)
    return flags[: n * w].reshape(n, w).mean(axis=1)


def calibrate(in_rates, shifted_rates, k_sd=6.0):
    half = len(in_rates) // 2
    cal, held = in_rates[:half], in_rates[half:]
    out = {}
    for name, thr in (("max_of_calibration_windows", float(cal.max())),
                      (f"mean_plus_{k_sd:g}sd", float(cal.mean() + k_sd * cal.std(ddof=1)))):
        out[name] = {
            "threshold": round(thr, 5),
            "false_alarm_rate_held_out": round(float((held > thr).mean()), 4),
            "false_alarms_held_out": f"{int((held > thr).sum())}/{len(held)}",
            "detection_rate_shifted": round(float((shifted_rates > thr).mean()), 4),
            "detections_shifted": f"{int((shifted_rates > thr).sum())}/{len(shifted_rates)}",
        }
    return out, {
        "windows_calibration": int(half), "windows_held_out": int(len(held)), "windows_shifted": int(len(shifted_rates)),
        "in_distribution_rate_min_median_max": [round(float(v), 4) for v in (in_rates.min(), np.median(in_rates), in_rates.max())],
        "shifted_rate_min_median_max": [round(float(v), 4) for v in (shifted_rates.min(), np.median(shifted_rates), shifted_rates.max())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dist", required=True, help="CICIDS2017 test_data.parquet")
    ap.add_argument("--shifted", required=True, help="CSE-CIC-IDS2018 test_data.parquet")
    ap.add_argument("--windows", default="500,2000,10000", help="window sizes for all traffic")
    ap.add_argument("--windows-forwarded", default="100,500,2000", help="window sizes for gate-forwarded flows")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp9_drift_alarm.json"))
    args = ap.parse_args()

    f32 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), "float32")
    bounds = margin_bounds(f32)
    first = f32.n_trees // 2 + 4
    schedule = [first] + [8] * ((f32.n_trees - first + 7) // 8)
    gate = Gate()

    flags = {}
    for name, path in (("in_distribution", args.in_dist), ("shifted", args.shifted)):
        x, _, _ = load_test_split(path)
        u = undecided_flags(f32, bounds, schedule, x)
        fwd = gate.score(x) >= gate.threshold
        flags[name] = {"all": u, "forwarded": u[fwd]}
        print(name, f"rows {len(x):,} undecided {u.mean():.4f} | forwarded {fwd.sum():,} undecided {u[fwd].mean():.4f}")

    report = {"first_check_trees": int(first), "note": __doc__.strip().splitlines()[-5:],
              "overall_undecided_rate": {k: {t: round(float(v[t].mean()), 4) for t in v} for k, v in flags.items()},
              "all_traffic": {}, "gate_forwarded": {}}
    for traffic, key, sizes in (("all_traffic", "all", args.windows), ("gate_forwarded", "forwarded", args.windows_forwarded)):
        for w in (int(s) for s in sizes.split(",")):
            in_rates = window_rates(flags["in_distribution"][key], w)
            sh_rates = window_rates(flags["shifted"][key], w)
            if len(in_rates) < 4 or len(sh_rates) < 1:
                continue
            thr, stats = calibrate(in_rates, sh_rates)
            report[traffic][str(w)] = {**stats, "thresholds": thr}
            print(f"  {traffic} W={w}: in-dist windows {stats['windows_calibration']}+{stats['windows_held_out']}, "
                  f"shifted {stats['windows_shifted']}; "
                  + "; ".join(f"{k}: thr {v['threshold']:.4f} FA {v['false_alarms_held_out']} det {v['detections_shifted']}" for k, v in thr.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
