"""Recovery of the recorded counts from a dark-subtracted rate (path B,
JSA section 3.4.1 and appendix B.1).

The output holds x_i = N_i/T_g - b_s only.  Two properties give N_i back:

1. x_i lies on a lattice of step q = 1/T_g shifted by the fraction of b_s
   below q, so T_g and that fraction follow from the values themselves;
   N0_i = T_g (x_i + b_s mod q) is an integer and N_i = N0_i + j_s with one
   integer j_s common to all points.
2. Counts are Poisson, so in a window without signal the sample variance
   equals the sample mean: j_s = Var(N0) - mean(N0).

The window is x <= Eth - margin with Eth from a forward fit, iterated until
the window no longer changes (start: the smallest j_s that keeps N >= 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import Spectrum


@dataclass
class RecoveryInfo:
    gate_time: float          # T_g = 1/q
    q: float                  # lattice step of the rate (cps)
    b_frac: float             # b_s mod q
    j_s: int                  # common integer offset of the counts
    j_min: int                # smallest j_s with all counts >= 0
    dark_rate: float          # b_s = b_frac + j_s q, the dark rate the instrument subtracted
    misfit: np.ndarray        # |T_g (x + b_frac) - nearest integer| per point
    bound: Optional[np.ndarray]   # rounding bound T_g u / 2 per point (None if the digits are unknown)
    integer_ok: bool          # all misfits within the bound
    window_upper: float       # upper end of the j_s window
    window_points: int
    iterations: int
    position: float           # Eth of the last fit inside the iteration

    def summary(self) -> str:
        b = "unknown" if self.bound is None else f"{self.bound.max():.2e}"
        return (f"T_g = {self.gate_time:g} s (q = {self.q:g}), b_s mod q = {self.b_frac:.6f}, "
                f"max misfit {self.misfit.max():.2e} (bound {b}) -> integer check {'passed' if self.integer_ok else 'FAILED'}; "
                f"j_s = {self.j_s} (min {self.j_min}), dark rate b_s = {self.dark_rate:.4f}; "
                f"window x <= {self.window_upper:g} ({self.window_points} points, {self.iterations} iterations)")


def default_candidates():
    """Lattice steps q = 1/T for gate times T = 0.5 ... 120 s, largest q first."""
    T = np.round(np.arange(0.5, 120.0001, 0.5), 6)
    return sorted({1.0 / t for t in T}, reverse=True)


def lattice_quantum(x, candidates=None, unit=None) -> float:
    """Largest candidate q such that all differences of the values are multiples of q."""
    xs = np.sort(np.asarray(x, float))
    d = np.diff(xs)
    if unit is not None:
        u = np.asarray(unit, float)[np.argsort(x)]
        tol = np.maximum(u[1:], u[:-1]) + 1e-12
    else:   # values printed with ~7 significant digits
        tol = 2e-6 * np.maximum(np.abs(xs[1:]), 1.0) + 1e-6
    for q in (default_candidates() if candidates is None else candidates):
        if np.all(np.abs(d - q * np.round(d / q)) < tol):
            return float(q)
    return float("nan")


def lattice_offset(x, q: float):
    """Common fractional offset b in x = q m - b (b in [0, q)) and the max misfit."""
    frac = np.mod(-np.asarray(x, float), q)
    # circular mean: values just below q and just above 0 belong to the same offset
    b = float(np.mod(q / (2 * np.pi) * np.angle(np.mean(np.exp(2j * np.pi * frac / q))), q))
    misfit = float(np.max(np.abs(((frac - b + q / 2) % q) - q / 2)))
    return b, misfit


def j_from_window(v, xw, j_min: int, estimator: str = "var") -> int:
    """j_s = Var(N0) - mean(N0) over the window (Eq. (B1)), rounded and kept >= j_min.
    estimator='detrend' uses the variance about a straight line instead."""
    v = np.asarray(v, float)
    if estimator == "detrend":
        res = v - np.polyval(np.polyfit(xw, v, 1), xw)
        var = float(np.sum(res ** 2) / (len(v) - 2))
    else:
        var = float(v.var(ddof=1))
    return int(max(j_min, round(var - float(v.mean()))))


def recover_counts(spec: Spectrum, model=None, xrange=None, margin: float = 0.35, gate_time: Optional[float] = None,
                   estimator: str = "var", max_iter: int = 6, min_window: int = 5, fit_kwargs: Optional[dict] = None,
                   dark_rate: Optional[float] = None):
    """Recover integer counts from a ``rate_dark_subtracted`` spectrum.

    Parameters
    ----------
    model
        forward model used to place the j_s window (default: PYS counting model, n = 2).
    xrange
        fitting range of those fits.
    margin
        the window is x <= Eth - margin (0.35 eV ~ 13 kT at room temperature for PYS).
    gate_time
        give T_g if it is known; otherwise it is found from the lattice.
    dark_rate
        give b_s if it is known (path A); then j_s is not estimated.

    Returns
    -------
    (Spectrum with signal_kind='counts', RecoveryInfo)
    """
    from .fit import fit
    from .presets import pys_counting

    if spec.signal_kind != "rate_dark_subtracted":
        raise ValueError("recover_counts expects signal_kind='rate_dark_subtracted'")
    x = spec.signal
    q = 1.0 / gate_time if gate_time else lattice_quantum(x, unit=spec.output_unit)
    if not np.isfinite(q):
        raise ValueError("the rates are not on a lattice; counts cannot be recovered (use path C)")
    Tg = 1.0 / q
    b, _ = lattice_offset(x, q)
    raw = Tg * (x + b)
    N0 = np.round(raw)
    misfit = np.abs(raw - N0)
    # rounding bound T_g u / 2 with u the coarsest last digit: the common fraction b is
    # estimated from all points, so it carries the rounding of the coarsest values too
    bound = None if spec.output_unit is None else np.full(len(x), Tg * float(spec.output_unit.max()) / 2 + 1e-9)
    ok = bool(np.all(misfit <= bound)) if bound is not None else bool(misfit.max() < 0.05)
    j_min = int(max(0, -N0.min()))

    def counts_spec(j):
        return spec.with_signal(N0 + j, "counts", integration_time=Tg, dark_rate=b + j * q)

    if dark_rate is not None:
        j = int(round((dark_rate - b) / q))
        info = RecoveryInfo(Tg, q, b, j, j_min, b + j * q, misfit, bound, ok, np.nan, 0, 0, np.nan)
        return counts_spec(j), info

    model = pys_counting() if model is None else model
    kw = dict(fit_kwargs or {})
    order = np.argsort(spec.x)
    j, prev, pos, it = j_min, None, float("nan"), 0
    win = np.zeros(spec.n_points, bool)
    for it in range(1, max_iter + 1):
        res = fit(counts_spec(j), model, xrange=xrange, **kw)
        pos = res.estimate
        win = spec.x <= pos - margin + 1e-9
        if win.sum() < min_window:
            win = np.zeros(spec.n_points, bool)
            win[order[:min_window]] = True
        if prev is not None and int(win.sum()) == prev:
            break
        prev = int(win.sum())
        j = j_from_window(N0[win], spec.x[win], j_min, estimator)
    info = RecoveryInfo(Tg, q, b, j, j_min, b + j * q, misfit, bound, ok, float(spec.x[win].max()), int(win.sum()), it, pos)
    return counts_spec(j), info
