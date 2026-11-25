"""Synthetic spectra with known truth, generated with the same equations as the
model (and with deliberate departures from it, to test the verdict).

:func:`counting_pys`
    lambda = d(x) + k Phi max(x - Eth, 0)^n, N ~ Poisson(T G^{-1}(lambda; tau_true));
    the instrument subtracts a dark rate b = G^{-1}(d) + delta and writes the
    dark-subtracted rate (as the AC series), optionally rounded to 7 digits.
:func:`current_pys` (experimental)
    I(t) = D + g(t) + k Phi relu^n, low-pass filtered, read at the end of each
    dwell, Gaussian noise sqrt(sigma0^2 + (rel I)^2).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .data import Spectrum
from .detector import DetectorCorrection, Identity, RikenAC
from .model import relu


def au_like_flux(x):
    """A smooth photon-flux shape similar to the AC-5 Au data (normalized to 1 at 5.9 eV).
    Replace it with a measured Phi when one is available."""
    x = np.asarray(x, float)
    p = 3.36 + 7.5 * np.clip((x - 3.8) / 2.0, 0, 1) ** 0.8 - 1.2 * np.clip((x - 5.8) / 0.4, 0, 1)
    return (p / x) / (np.interp(5.9, x, p) / 5.9) if x.min() <= 5.9 <= x.max() else p / p.max()


def counting_pys(x=None, phi=None, Eth: float = 4.95, n: float = 2.0, k: float = 192.0, d: float = 5.0,
                 delta: float = 0.0, T: float = 4.0, tau: float = 0.00457, tau_true: Optional[float] = None,
                 background=None, rng=None, digits: Optional[int] = 7, output: str = "rate_dark_subtracted"):
    """Synthetic AC-type PYS spectrum.

    Parameters
    ----------
    d, delta
        true background rate (before losses) and the excess of the subtracted
        dark rate over the recorded background rate, delta = b - G^{-1}(d).
    tau, tau_true
        dead time written in the header (used by the analysis) and the true one.
    background
        callable x -> background rate replacing the constant d (e.g. a decaying one).
    output
        'rate_dark_subtracted' (AC series), or 'counts'.

    Returns
    -------
    (Spectrum, truth dict)
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.round(np.arange(3.80, 6.2001, 0.05), 2) if x is None else np.asarray(x, float)
    phi = au_like_flux(x) if phi is None else np.asarray(phi, float)
    det = RikenAC(tau)
    det_true = RikenAC(tau if tau_true is None else tau_true)
    bg = np.full_like(x, d) if background is None else np.asarray(background(x), float)
    lam = bg + k * phi * relu(x - Eth) ** n
    mu = T * det_true.G_inv(lam)
    N = rng.poisson(mu).astype(float)
    J = int(round(T * (float(det_true.G_inv(np.array(bg.mean()))) + delta)))
    b = J / T
    truth = dict(Eth=Eth, n=n, k=k, d=d, delta=delta, dark_rate=b, counts=N, mu=mu, lam=lam, T=T)
    if output == "counts":
        return Spectrum(x, N, phi, "counts", integration_time=T, dark_rate=b, detector=det,
                        meta=dict(synthetic=True)), truth
    rate = (N - J) / T
    unit = None
    if digits is not None:
        rate = np.array([float(f"{v:.{digits}g}") for v in rate])
        unit = np.array([10.0 ** (np.floor(np.log10(abs(v))) - digits + 1) if v != 0 else 10.0 ** -digits
                         for v in rate])
    return Spectrum(x, rate, phi, "rate_dark_subtracted", detector=det, output_unit=unit,
                    output=det.G(rate) / phi, meta=dict(synthetic=True, header_gate_time=T)), truth


def current_pys(x=None, phi=None, Eth: float = 5.0, n: float = 2.0, I_max: float = 20.0, D: float = 0.05,
                sigma0: float = 0.01, rel: float = 0.01, dwell: float = 5.0, t_start: float = 30.0,
                tau_f: float = 0.5, relax: str = "none", amp: float = 1.0, tau: float = 60.0, m: float = 0.8,
                rng=None, dt: float = 0.05):
    """Synthetic current-method PYS sweep (experimental).  Units: pA, s.
    relax: 'none' | 'exp' (A exp(-(t - t_start)/tau)) | 'power' (A (t/t_start)^-m)."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.round(np.arange(4.0, 7.0001, 0.05), 2) if x is None else np.asarray(x, float)
    phi = au_like_flux(x) if phi is None else np.asarray(phi, float)
    k = I_max / (phi[np.argmax(x)] * relu(x.max() - Eth) ** n)
    N = len(x)
    nf = int(round(N * dwell / dt))
    tf = t_start + (np.arange(nf) + 1) * dt
    idx = np.minimum(((tf - t_start - 1e-9) // dwell).astype(int), N - 1)

    def g(t):
        if relax == "exp":
            return amp * np.exp(-(t - t_start) / tau)
        if relax == "power":
            return amp * (t / t_start) ** (-m)
        return np.zeros_like(t)

    xin = D + g(tf) + k * phi[idx] * relu(x[idx] - Eth) ** n
    a = 1.0 - np.exp(-dt / tau_f)
    y = np.empty_like(xin)
    acc = xin[0]
    for i in range(nf):
        acc += (xin[i] - acc) * a
        y[i] = acc
    read = np.minimum([np.searchsorted(tf, t_start + (j + 1) * dwell - 1e-9) for j in range(N)], nf - 1)
    t_read = tf[read]
    clean = y[read]
    I = clean + rng.normal(0.0, np.sqrt(sigma0 ** 2 + (rel * clean) ** 2))
    truth = dict(Eth=Eth, n=n, k=k, D=D, background=D + g(t_read), clean=clean, sigma0=sigma0, rel=rel)
    return Spectrum(x, I, phi, "current", time=t_read, detector=Identity(),
                    meta=dict(synthetic=True, unit="pA")), truth
