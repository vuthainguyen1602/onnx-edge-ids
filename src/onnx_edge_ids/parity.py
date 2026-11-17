#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Label comparison for the release gate, with exact score ties set apart.

Routing is fixed by IEEE 754, but the label of a row whose class scores tie
depends on the order in which leaf weights are summed, and ONNX Runtime's
order changes with its intra-op thread count. A label difference where both
runtimes put P(class 1) within ``band`` of one half is therefore
indeterminate, not evidence of a fault; every other label difference is.
"""

import numpy as np


def p_class1(preds, confidences):
    """P(class 1) from a binary engine's predicted label and its confidence."""
    preds = np.asarray(preds)
    conf = np.asarray(confidences, dtype=np.float64)
    return np.where(preds == 1, conf, 1.0 - conf)


def split_label_disagreements(pred_a, conf_a, pred_b, conf_b, band=0.0):
    """Boolean masks (failing, indeterminate) over rows whose labels differ.

    With ``band`` 0 every label difference fails, the gate's default.
    """
    differ = np.asarray(pred_a) != np.asarray(pred_b)
    if band <= 0:
        return differ, np.zeros_like(differ)
    tie = (np.abs(p_class1(pred_a, conf_a) - 0.5) <= band) & (np.abs(p_class1(pred_b, conf_b) - 0.5) <= band)
    return differ & ~tie, differ & tie
