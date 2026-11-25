"""Generic input: arrays or a delimited text file with named columns.

Use this for instruments without a dedicated reader.  Describe what the
instrument did (detector correction, subtracted dark rate) in the arguments,
so that the corrections can be put into the model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data import Spectrum
from ..detector import DetectorCorrection, Identity


def from_arrays(x, signal, reference=None, kind: str = "counts", integration_time=None, time=None,
                dark_rate: Optional[float] = None, detector: Optional[DetectorCorrection] = None, **meta) -> Spectrum:
    return Spectrum(x=x, signal=signal, reference=reference, signal_kind=kind, integration_time=integration_time,
                    time=time, dark_rate=dark_rate, detector=Identity() if detector is None else detector, meta=meta)


def read_table(path: str, x: str, signal: str, reference: Optional[str] = None, time: Optional[str] = None,
               kind: str = "counts", delimiter: Optional[str] = ",", skiprows: int = 0, **kw) -> Spectrum:
    """Read columns by header name from a delimited text file (first non-skipped line = header)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()[skiprows:]
    head = [h.strip() for h in (lines[0].split(delimiter) if delimiter else lines[0].split())]
    arr = np.array([[float(v) for v in (ln.split(delimiter) if delimiter else ln.split())]
                    for ln in lines[1:] if ln.strip()])
    col = {h: arr[:, i] for i, h in enumerate(head)}
    return from_arrays(col[x], col[signal], None if reference is None else col[reference], kind,
                       time=None if time is None else col[time], file=path, **kw)
