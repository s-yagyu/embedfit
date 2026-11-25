"""Reader for the Riken Keiki AC series ``.dat`` files (AC-5, AC-2S, AC-3, ...).

File layout (one scan per file)::

    line 1  fileType, deadTime, countingTime, powerNumber, anodeVoltage, step, model,
            yAxisMaximum, startEnergy, finishEnergy[, flagDifDataGroundLevel, bgCountingRate]
    line 2  measureDate, sampleName
    line 3  uvIntensity59, targetUv, nameLightCorrection[, sensitivity1, sensitivity2]
    rows    energy, countingRate, background flag, rise flag, uvIntensity

``countingRate`` is the recorded rate after the dark rate b_s was subtracted
(b_s itself is not written).  The instrument then computes
Y = G(x)/Phi with G of :class:`~embedfit.detector.RikenAC` and
Phi = (P/E)/(P_5.9/5.9).

For AC-2 and AC-3 the instrument applies no counting correction (the vendor
converter uses G(m) = m); the reader follows the instrument and returns
:class:`~embedfit.detector.Identity` for them, with a note in ``meta``.

The gate time is not taken from the header: in the AC series it is the header
counting time minus 1 s (manufacturer), and :func:`embedfit.recover.recover_counts`
finds it from the lattice of the rates.
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from ..data import Spectrum
from ..detector import Identity, RikenAC

_ENCODINGS = ("utf-8", "shift_jis", "euc-jp", "iso-2022-jp")


def _read_rows(path):
    for enc in _ENCODINGS:
        try:
            with open(path, encoding=enc, newline="") as f:
                return [row for row in csv.reader(f)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _unit(s: str) -> float:
    """Unit of the last printed digit of a number written as text."""
    s = s.strip().lower()
    mant, _, exp = s.partition("e")
    e = int(exp) if exp else 0
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (e - dec)


def photon_flux(P, E, P59):
    """Phi = (P/E)/(P_5.9/5.9): light power converted to a photon number, 1 at 5.9 eV."""
    return (np.asarray(P, float) / np.asarray(E, float)) / (P59 / 5.9)


def read_ac(path: str) -> Spectrum:
    rows = _read_rows(path)
    h0, h1, h2 = rows[0], rows[1], rows[2]
    if len(h0) == 10:
        h0 = h0 + ["0", "0.0"]
    if len(h2) == 3:
        h2 = h2 + ["1", "1"]
    data = [r for r in rows[3:] if len(r) >= 5 and r[0].strip()]
    E = np.array([float(r[0]) for r in data])
    x = np.array([float(r[1]) for r in data])
    P = np.array([float(r[4]) for r in data])
    unit = np.array([_unit(r[1]) for r in data])
    model = h0[6].strip()
    tau, sens = float(h0[1]), float(h2[3])
    P59 = float(h2[0])
    meta = dict(file=os.path.basename(path), instrument=model, dead_time=tau, header_counting_time=float(h0[2]),
                power_number=float(h0[3]), anode_voltage=float(h0[4]), step=float(h0[5]),
                bg_counting_rate_header=float(h0[11]), date=h1[0].strip(), sample=h1[1].strip() if len(h1) > 1 else "",
                P59=P59, target_power=float(h2[1]), light_correction_file=h2[2].strip(), sensitivity1=sens,
                sensitivity2=float(h2[4]), bg_flag=np.array([int(float(r[2])) for r in data]),
                rise_flag=np.array([int(float(r[3])) for r in data]), light_power=P)
    if model in ("AC-2", "AC-3"):
        det = Identity()
        meta["note"] = "the instrument applies no counting correction (G(m) = m) for this model"
    else:
        det = RikenAC(tau=tau, sens=sens)
    phi = photon_flux(P, E, P59)
    m = re.search(r"_(\d+)\.dat$", os.path.basename(path))
    label = f"{meta['sample']} scan {m.group(1)}" if m else meta["sample"]
    return Spectrum(x=E, signal=x, reference=phi, signal_kind="rate_dark_subtracted", detector=det,
                    output=det.G(x) / phi, output_unit=unit, meta=meta, label=label)


def read_ac_folder(folder: str, pattern: str = "*_*.dat") -> list:
    """All scans of a folder (``<name>_<scan>.dat``) sorted by scan number."""
    files = glob.glob(os.path.join(folder, pattern))

    def idx(p):
        m = re.search(r"_(\d+)\.dat$", p)
        return int(m.group(1)) if m else 0

    return [read_ac(f) for f in sorted(files, key=idx)]
