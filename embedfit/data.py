"""Measured data and what the instrument did to it.

A :class:`Spectrum` keeps the recorded signal apart from everything that was
used to turn it into the instrument's final output: the reference quantity
(photon flux, I0, ...), the integration time, the subtracted dark level, the
detector correction G and the final output itself.  Keeping them apart is what
lets the corrections be applied to the *model* instead of to the data.

Signal kinds
------------
``"counts"``
    integer counts per point (Poisson).  ``integration_time`` is required.
``"rate"``
    recorded count rate N/T (dark not subtracted).
``"rate_dark_subtracted"``
    recorded rate minus a dark rate b that is not stored (AC series).  Use
    :func:`embedfit.recover.recover_counts` to go back to counts (path B).
``"current"``
    a current or any other continuous reading (Gaussian noise model).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from .detector import DetectorCorrection, Identity

SIGNAL_KINDS = ("counts", "rate", "rate_dark_subtracted", "current")


@dataclass
class Spectrum:
    """One sweep (scan) of a spectrum.

    Parameters
    ----------
    x
        Swept variable (photon energy in eV for PYS), ascending or not.
    signal
        The recorded signal, of kind ``signal_kind``.
    reference
        Reference quantity Phi(x) by which the instrument normalizes the signal
        (dimensionless or in any fixed unit).  It enters the model as a known
        input.  Ones if the measurement has no reference channel.
    signal_kind
        One of :data:`SIGNAL_KINDS`.
    integration_time
        Integration (gate) time per point, scalar or array, in s.  Required
        for counts; for ``rate_dark_subtracted`` it may be left ``None`` and
        found from the data (path B).
    time
        Time of each reading since the start of the sweep (s).  Needed only for
        time-dependent backgrounds (relaxation, drift).
    dark_rate
        Dark rate subtracted by the instrument, if it is known.
    detector
        Detector correction G that the instrument applies (maps the recorded
        rate to the true rate).  :class:`~embedfit.detector.Identity` if none.
    output
        The instrument's final output (e.g. the photoelectron yield), kept for
        comparison and plotting only; the analysis never fits it.
    output_unit
        Unit of the last printed digit of each recorded value (array), used to
        bound the rounding error when counts are recovered.
    meta
        Free-form header information (instrument, sample, date, file name...).
    """

    x: np.ndarray
    signal: np.ndarray
    reference: Optional[np.ndarray] = None
    signal_kind: str = "counts"
    integration_time: Any = None
    time: Optional[np.ndarray] = None
    dark_rate: Optional[float] = None
    detector: DetectorCorrection = field(default_factory=Identity)
    output: Optional[np.ndarray] = None
    output_unit: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, float)
        self.signal = np.asarray(self.signal, float)
        n = len(self.x)
        if self.signal.shape != (n,):
            raise ValueError("signal must have the same length as x")
        self.reference = np.ones(n) if self.reference is None else np.asarray(self.reference, float)
        if self.reference.shape != (n,):
            raise ValueError("reference must have the same length as x")
        if self.signal_kind not in SIGNAL_KINDS:
            raise ValueError(f"signal_kind must be one of {SIGNAL_KINDS}")
        if self.time is not None:
            self.time = np.asarray(self.time, float)
        if self.output is not None:
            self.output = np.asarray(self.output, float)
        if self.output_unit is not None:
            self.output_unit = np.broadcast_to(np.asarray(self.output_unit, float), (n,)).copy()
        if self.signal_kind == "counts" and self.integration_time is None:
            raise ValueError("counts need an integration_time")

    # ------------------------------------------------------------------ #
    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def is_counting(self) -> bool:
        return self.signal_kind in ("counts", "rate", "rate_dark_subtracted")

    def T(self) -> np.ndarray:
        """Integration time as an array (ones if not given)."""
        if self.integration_time is None:
            return np.ones(self.n_points)
        return np.broadcast_to(np.asarray(self.integration_time, float), (self.n_points,)).copy()

    def counts(self) -> np.ndarray:
        """Recorded counts N_i (only for ``counts`` and ``rate`` data)."""
        if self.signal_kind == "counts":
            return self.signal
        if self.signal_kind == "rate":
            return np.round(self.signal * self.T())
        raise ValueError(f"no counts in a '{self.signal_kind}' spectrum; recover them first")

    def recorded_rate(self) -> np.ndarray:
        """Recorded rate including the dark level (counts/T, or the current)."""
        if self.signal_kind == "counts":
            return self.signal / self.T()
        if self.signal_kind == "rate":
            return self.signal
        if self.signal_kind == "rate_dark_subtracted":
            if self.dark_rate is None:
                raise ValueError("dark rate unknown; recover counts first")
            return self.signal + self.dark_rate
        return self.signal

    def corrected_value(self, dark_rate: Optional[float] = None) -> np.ndarray:
        """Data-correction route, Eq. (3): y = G(N/T - b) / Phi (no clipping).

        For ``rate_dark_subtracted`` data the stored rate is used as it is
        (the instrument already subtracted b).
        """
        if self.signal_kind == "rate_dark_subtracted":
            m = self.signal
        else:
            b = self.dark_rate if dark_rate is None else dark_rate
            m = self.recorded_rate() - (0.0 if b is None else b)
        return self.detector.G(m) / self.reference

    def subset(self, mask) -> "Spectrum":
        """Copy restricted to the points selected by a boolean mask."""
        mask = np.asarray(mask, bool)

        def cut(a):
            if a is None:
                return None
            a = np.asarray(a)
            return a[mask] if a.ndim == 1 and len(a) == len(mask) else a

        return replace(self, x=self.x[mask], signal=self.signal[mask], reference=self.reference[mask],
                       integration_time=cut(self.integration_time), time=cut(self.time),
                       output=cut(self.output), output_unit=cut(self.output_unit), meta=dict(self.meta))

    def with_signal(self, signal, kind: str, **changes) -> "Spectrum":
        """Copy with a new signal (used when counts are recovered)."""
        return replace(self, signal=np.asarray(signal, float), signal_kind=kind, meta=dict(self.meta), **changes)
