"""Data-correction methods, kept for comparison with the model-embedded fit.

* :func:`yield_fit` -- fit of the corrected value (no clipping) with a
  background of the 1/Phi form, (d - b)/Phi (JSA Eq. (5) with c = 0; the
  "yield fit, 1/Phi form" of the paper), unweighted or weighted.
* :func:`path_c` -- the same fit with weights from Eq. (6) iterated with the
  fitted values (path C, appendix B.2), for outputs whose counts cannot be
  recovered.
* :func:`crossing` -- the conventional crossing point of a horizontal line and
  a straight line in Y^(1/n) after zero clipping, over fixed windows.

The automated conventional method of the JSA paper (1/n-Scan) is published
separately (n-pys) and is not re-implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .data import Spectrum
from .fit import range_mask
from .model import relu


@dataclass
class YieldFit:
    Eth: float
    se: float
    D: float
    k: float
    chi2: float
    dof: int
    iterations: int = 0


def yield_fit(x, y, phi, n: float = 2.0, sigma=None, background: str = "1/phi", Eth0: Optional[float] = None,
              offsets=(-0.2, 0.0, 0.2)) -> YieldFit:
    """Least squares of y = D/Phi + k max(x - Eth, 0)^n (background='1/phi'; D is the
    residual d - b of the dark subtraction) or y = D + k max(...)^n (background='const').
    'd/phi' is accepted as an alias of '1/phi'.  sigma=None: unweighted, and the
    SE is scaled by sqrt(chi2/dof)."""
    if background not in ("1/phi", "d/phi", "const"):
        raise ValueError(f"background must be '1/phi' or 'const', not {background!r}")
    x, y, phi = (np.asarray(a, float) for a in (x, y, phi))
    s = np.ones_like(y) if sigma is None else np.asarray(sigma, float)
    basis = (1.0 / phi) if background in ("1/phi", "d/phi") else np.ones_like(phi)
    Eth0 = float(np.median(x)) if Eth0 is None else Eth0

    def res(p):
        return (p[0] * basis + np.exp(p[1]) * relu(x - p[2]) ** n - y) / s

    best = None
    k0 = max(y.max(), 1.0) / max(relu(x.max() - Eth0) ** n, 1e-9)
    for off in offsets:
        r = least_squares(res, [0.0, np.log(k0), Eth0 + off], x_scale="jac", max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    cov = np.linalg.pinv(J.T @ J)
    chi2, dof = float(2 * best.cost), len(x) - 3
    se = float(np.sqrt(cov[2, 2]))
    if sigma is None:
        se *= np.sqrt(chi2 / dof)
    return YieldFit(float(best.x[2]), se, float(best.x[0]), float(np.exp(best.x[1])), chi2, dof)


def path_c(spec: Spectrum, n: float = 2.0, xrange=None, dark_rate: Optional[float] = None, tol: float = 1e-5,
           max_iter: int = 50, Eth0: Optional[float] = None) -> YieldFit:
    """Path C (appendix B.2): fit the corrected value with weights
    sigma_y = G'(m) sqrt(mu) / (T Phi), mu = T {G^{-1}(Phi y_hat) + b}, iterated until
    Eth changes by less than `tol` (in x units).  Needs G, T and the dark rate b."""
    b = spec.dark_rate if dark_rate is None else dark_rate
    if b is None or spec.integration_time is None:
        raise ValueError("path C needs the dark rate and the integration time")
    m = range_mask(spec.x, xrange)
    x, phi, T = spec.x[m], spec.reference[m], spec.T()[m]
    y = spec.corrected_value(b)[m]
    det = spec.detector
    r = yield_fit(x, y, phi, n, None, Eth0=Eth0)
    for it in range(1, max_iter + 1):
        yhat = r.D / phi + r.k * relu(x - r.Eth) ** n
        mu = np.maximum(T * (det.G_inv(phi * yhat) + b), 1.0)
        # same approximation as Eq. (6) and App. B.2 of the paper: G' is taken at the recorded rate
        sig = det.dG(mu / T) * np.sqrt(mu) / (T * phi)
        new = yield_fit(x, y, phi, n, sig, Eth0=r.Eth)
        done = abs(new.Eth - r.Eth) < tol
        r = new
        if done:
            break
    r.iterations = it
    return r


def crossing(x, y, n: float, bg_window, rise_window, clip: bool = True) -> dict:
    """Conventional threshold: s = max(y, 0)^(1/n); horizontal line = mean of s over
    bg_window; straight line over rise_window; Eth = crossing point."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = (np.clip(y, 0, None) if clip else y) ** (1.0 / n)
    bm = (x >= bg_window[0]) & (x <= bg_window[1])
    rm = (x >= rise_window[0]) & (x <= rise_window[1])
    bg = float(np.mean(s[bm]))
    a, c = np.polyfit(x[rm], s[rm], 1)
    return dict(Eth=float((bg - c) / a), bg=bg, slope=float(a), intercept=float(c), s=s)
