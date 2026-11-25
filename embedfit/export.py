"""CSV output of the numbers drawn in the figures (no pandas needed)."""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

import numpy as np

from .fit import evaluate


def _write(path, header, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in zip(*cols):
            w.writerow([f"{v:.10g}" if isinstance(v, (float, np.floating)) else v for v in row])
    return path


def analysis_to_csv(an, folder: str, stem: Optional[str] = None) -> list:
    """Writes <stem>_points.csv (per point) and <stem>_report.json (values to report)."""
    os.makedirs(folder, exist_ok=True)
    stem = stem or (an.input.label or "spectrum").replace(" ", "_")
    sp, r, dy = an.spectrum, an.result, an.display_
    ev = evaluate(r, sp)
    n = sp.n_points
    in_fit = r.mask.astype(int)
    resid = np.full(n, np.nan)
    resid[r.mask] = r.r
    rec = sp.counts() if sp.is_counting else sp.signal
    cols = [sp.x, sp.reference, sp.T(), rec, ev["mu"], ev["lam"], ev["background"], ev["kf"], in_fit, resid]
    head = ["x", "phi", "T", "recorded", "mu", "lambda", "background", "kf", "in_fit", "std_residual"]
    if dy is not None:
        cols += [dy.y, dy.sigma, dy.d_hat]
        head += ["display_yield", "sigma_y", "d_hat"]
        if dy.instrument is not None:
            cols.append(dy.instrument)
            head.append("instrument_value")
    p1 = _write(os.path.join(folder, f"{stem}_points.csv"), head, cols)
    p2 = os.path.join(folder, f"{stem}_report.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(an.report(), f, indent=2, ensure_ascii=False, default=float)
    return [p1, p2]


def scans_to_csv(ss, path: str) -> str:
    rows = ss.table()
    keys = list(rows[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return _write(path, keys, [[row[k] for row in rows] for k in keys])
