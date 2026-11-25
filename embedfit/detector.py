"""Detector corrections G: recorded rate m -> true rate G(m).

The instrument applies G to the data; the model-embedded method applies its
inverse to the model instead: the expected recorded rate is G^{-1}(lambda).
A correction only has to be monotonically increasing on the range used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DetectorCorrection:
    """Base class.  Subclasses implement :meth:`G` and :meth:`m_max`."""

    name = "detector"

    def G(self, m):
        raise NotImplementedError

    def m_max(self) -> float:
        """Upper end of the recorded-rate range on which G is increasing."""
        return np.inf

    def dG(self, m, h: float = 1e-4):
        m = np.asarray(m, float)
        return (self.G(m + h) - self.G(m - h)) / (2 * h)

    def G_inv(self, lam, n_iter: int = 80):
        """Recorded rate m with G(m) = lam (vectorized bisection on [0, m_max)).

        Negative lam is mapped linearly with the slope of G at 0, so that a fit
        can pass through slightly negative backgrounds without breaking.
        """
        lam = np.asarray(lam, float)
        top = self.m_max()
        if not np.isfinite(top):
            top = max(1.0, float(np.max(np.abs(lam)))) * 4.0
            while np.any(self.G(top) < lam):
                top *= 4.0
        top *= 1 - 1e-9
        lo = np.zeros_like(lam)
        hi = np.full_like(lam, top)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            up = self.G(mid) < lam
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        slope0 = float(self.G(np.array(1e-9))) / 1e-9
        return np.where(lam <= 0, lam / slope0, 0.5 * (lo + hi))

    def describe(self) -> str:
        return self.name


@dataclass
class Identity(DetectorCorrection):
    """No correction (current measurements, or instruments that apply none)."""

    name = "identity"

    def G(self, m):
        return np.asarray(m, float)

    def dG(self, m, h: float = 1e-4):
        return np.ones_like(np.asarray(m, float))

    def G_inv(self, lam, n_iter: int = 80):
        return np.asarray(lam, float)


@dataclass
class NonParalyzable(DetectorCorrection):
    """Non-paralyzable dead time: G(m) = m / (1 - tau m)."""

    tau: float
    name = "non-paralyzable dead time"

    def G(self, m):
        m = np.asarray(m, float)
        return m / (1.0 - self.tau * m)

    def m_max(self) -> float:
        return 1.0 / self.tau

    def G_inv(self, lam, n_iter: int = 80):
        lam = np.asarray(lam, float)
        return lam / (1.0 + self.tau * lam)

    def describe(self) -> str:
        return f"G(m) = m/(1 - tau m), tau = {self.tau:g} s"


@dataclass
class RikenAC(DetectorCorrection):
    """Correction of the Riken Keiki AC series (AC-5, AC-2S, ...), JSA Eq. (11)::

        G(m) = m / (1 - tau m) * exp(a / (1 - beta m)) * sens

    tau is the dead time in the file header, a = 0.13571 and beta = 0.0028 are
    the constants of the rate-dependent efficiency correction, sens is the
    sensitivity factor in the header.
    """

    tau: float
    sens: float = 1.0
    a: float = 0.13571
    beta: float = 0.0028
    name = "Riken Keiki AC"

    def G(self, m):
        m = np.asarray(m, float)
        return m / (1.0 - self.tau * m) * np.exp(self.a / (1.0 - self.beta * m)) * self.sens

    def m_max(self) -> float:
        return min(1.0 / self.tau, 1.0 / self.beta)

    def describe(self) -> str:
        return (f"G(m) = m/(1 - tau m) exp(a/(1 - beta m)) sens, tau = {self.tau:g} s, "
                f"a = {self.a:g}, beta = {self.beta:g}, sens = {self.sens:g}")
