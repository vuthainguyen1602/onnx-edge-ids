"""Release-gate checks: a replay gate (labels, or labels and confidences) and a probe gate."""

import numpy as np

from edgeids.runtime import label

TOLERANCES = (1e-5, 1e-4, 1e-3, 1e-2)


def replay_gate(p, ref, replay_rows):
    delta = np.abs(p - ref).max(axis=1)
    flips = label(p) != label(ref)
    out = {"label_flips": int(flips.sum()), "max_delta": float(delta.max()),
           "prob_replay_passes_labels_only": float((1.0 - flips.mean()) ** replay_rows), "tolerances": {}}
    for tol in TOLERANCES:
        trips = flips | (delta > tol)
        out["tolerances"][f"{tol:g}"] = {"rows_tripping_gate": int(trips.sum()),
                                         "prob_replay_passes": float((1.0 - trips.mean()) ** replay_rows)}
    return out


TOL = 1e-5


def probe_gate(p, ref):
    delta = np.abs(p - ref).max(axis=1)
    flips = label(p) != label(ref)
    over = delta > TOL
    return {"label_flips": int(flips.sum()), "rows_over_tolerance": int(over.sum()),
            "max_delta": float(delta.max()), "gate_fails": bool((flips | over).any())}
