"""Display yield (JSA section 4.3, Eq. (14)): a corrected spectrum for figures
that keeps the noise distribution and has a flat background.

    counting:  y_disp_i = (G(N_i/T_i) - d_hat) / Phi_i
    current :  y_disp_i = (I_i - d_hat(t_i)) / Phi_i

The level d_hat replaces the dark level b subtracted by the instrument.  It is
taken either from a forward fit (``method="fit"``) or from a window without
signal only (``method="window"``, appendix B.4), so that the display does not
depend on the response f at all.  G is applied to the recorded rate including
the dark level, as in the forward model.  The expectation of y_disp is
c + k f(x); its sd is sigma_y,i of JSA Eq. (6).  The uncertainty of d_hat adds
a common shift +-sigma_d/Phi_i.

Do not take y_disp^(1/n) of noisy data: its variance diverges near 0.  For
Y^(1/n) plots raise only the model curve to 1/n (see :mod:`embedfit.plot`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .data import Spectrum
from .fit import FitResult, evaluate, onset


@dataclass
class DisplayYield:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    d_hat: np.ndarray           # level subtracted at each point (constant for a constant background)
    d_se: float
    method: str                 # 'fit' | 'window' | 'given'
    window: Optional[np.ndarray] = None     # points of the signal-free window (method='window')
    model: Optional[np.ndarray] = None      # expectation c + k f on the same scale (if a fit is given)
    kf: Optional[np.ndarray] = None
    instrument: Optional[np.ndarray] = None  # corrected value with the instrument's dark level, Eq. (3)

    def flat_check(self, mask=None) -> dict:
        """Mean and spread of y/sigma in the signal-free window (expect ~0 and ~1)."""
        m = self.window if mask is None else np.asarray(mask, bool)
        if m is None:
            raise ValueError("no window to check")
        z = self.y[m] / self.sigma[m]
        return dict(points=int(m.sum()), mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)),
                    negative=int(np.sum(self.y[m] < 0)))


def _window_level(spec: Spectrum, n_low: int, k_sigma: float, run: int, gap: Optional[float]):
    """Level of the signal-free window, appendix B.4.  Returns (d_hat, se, window)."""
    det = spec.detector
    if spec.is_counting:
        N = spec.counts()
        T = spec.T()
        _, win = onset(spec.x, N / T, np.sqrt(np.maximum(N, 1.0)) / T, n_low, k_sigma, run, gap)
        Nbar, Tw = float(N[win].mean()), float(T[win].mean())
        m = Nbar / Tw
        d = float(det.G(np.array(m)))
        se = float(det.dG(np.array(m))) * np.sqrt(Nbar / win.sum()) / Tw
        return d, se, win
    I = spec.signal
    sd = np.full(len(I), np.std(I[np.argsort(spec.x)[:n_low]], ddof=1))
    _, win = onset(spec.x, I, sd, n_low, k_sigma, run, gap)
    return float(I[win].mean()), float(I[win].std(ddof=1) / np.sqrt(win.sum())), win


def display_yield(spec: Optional[Spectrum] = None, result: Optional[FitResult] = None,
                  method: Union[str, float] = "fit", n_low: int = 5, k_sigma: float = 3.0, run: int = 2,
                  gap: Optional[float] = None, noise=None) -> DisplayYield:
    """Display yield of `spec` (default: the spectrum of `result`).

    method
        ``"fit"``: d_hat from the background terms of the fit that do not follow
        Phi; ``"window"``: from the signal-free window only (no response f);
        a number: use it as d_hat.
    gap
        distance kept below the onset for the window (0.3 eV in the JSA paper).
    """
    if spec is None:
        if result is None:
            raise ValueError("give a spectrum or a fit result")
        spec = result.spectrum
    det = spec.detector
    phi = spec.reference
    counting = spec.is_counting
    window, model_y, kf = None, None, None
    ev = evaluate(result, spec) if result is not None else None

    if isinstance(method, (int, float)):
        d_hat, d_se, how = np.full(spec.n_points, float(method)), 0.0, "given"
    elif method == "fit":
        if result is None:
            raise ValueError("method='fit' needs a fit result")
        d_hat = ev["background_const"]
        names = [b.name for b in result.model.background if hasattr(b, "name") and not b.flux_proportional]
        # d_se is the SE of the single named flux-independent term; a time-dependent term without
        # a name (e.g. ExpDecay in the experimental current mode) enters d_hat but not d_se
        d_se = float(result.se.get(names[0], 0.0)) if len(names) == 1 else float("nan")
        how = "fit"
    elif method == "window":
        d, d_se, window = _window_level(spec, n_low, k_sigma, run, gap)
        d_hat, how = np.full(spec.n_points, d), "window"
    else:
        raise ValueError("method must be 'fit', 'window' or a number")

    if counting:
        N = spec.counts()
        T = spec.T()
        y = (det.G(N / T) - d_hat) / phi
        mu = ev["mu"] if ev is not None else np.maximum(N, 1.0)
        sigma = det.dG(mu / T) * np.sqrt(np.maximum(mu, 1e-9)) / (T * phi)
        instrument = spec.corrected_value() if spec.dark_rate is not None else None
    else:
        I = spec.signal
        y = (I - d_hat) / phi
        nm = noise if noise is not None else (result.model.noise if result is not None else None)
        if nm is None:
            raise ValueError("a current spectrum needs a noise model (or a fit result)")
        sigma = nm.sd(ev["mu"] if ev is not None else I) / phi
        instrument = None
    if ev is not None:
        kf = ev["kf"]
        model_y = kf + (ev["background"] - ev["background_const"]) / phi     # c + k f
    return DisplayYield(spec.x, y, sigma, d_hat, d_se, how, window, model_y, kf, instrument)
