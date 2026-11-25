"""Noise models: how the recorded value scatters around its expectation mu.

Poisson
    counts (JSA Eq. (7)); the variance equals the mean, so no noise parameter
    is left and the standard error follows from one spectrum.
Gaussian
    continuous readings such as a current: sd(mu) = sqrt(sigma0^2 + (rel mu)^2).
    sigma0 and rel are inputs (measured separately, e.g. from a dark sweep).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TINY = 1e-9


class NoiseModel:
    counting = False

    def sd(self, mu):
        raise NotImplementedError

    def fit_residuals(self, y, mu, sd_fixed=None):
        """Residuals whose sum of squares is minimized (equivalent to the MLE)."""
        raise NotImplementedError

    def pearson(self, y, mu):
        """Standardized residuals r_i = (y_i - mu_i) / sd(mu_i)."""
        return (y - mu) / self.sd(mu)

    def deviance(self, y, mu) -> float:
        """-2 log L up to a constant that does not depend on the model."""
        raise NotImplementedError

    def fisher_weights(self, mu):
        return 1.0 / self.sd(mu) ** 2

    def describe(self) -> str:
        return type(self).__name__


@dataclass
class Poisson(NoiseModel):
    counting = True

    def sd(self, mu):
        return np.sqrt(np.maximum(mu, _TINY))

    def fit_residuals(self, y, mu, sd_fixed=None):
        mu = np.maximum(mu, _TINY)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu)
        return np.sign(y - mu) * np.sqrt(2 * np.maximum(t, 0))

    def deviance(self, y, mu) -> float:
        """Poisson deviance, JSA Eq. (9)."""
        mu = np.maximum(mu, _TINY)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu)
        return float(2 * np.sum(t))

    def describe(self) -> str:
        return "Poisson (variance = mean)"


@dataclass
class Gaussian(NoiseModel):
    """sd = sqrt(sigma0^2 + (rel mu)^2), evaluated at the model (IRLS)."""

    sigma0: float = 1.0
    rel: float = 0.0

    def sd(self, mu):
        mu = np.asarray(mu, float)
        return np.sqrt(self.sigma0 ** 2 + (self.rel * mu) ** 2)

    def fit_residuals(self, y, mu, sd_fixed=None):
        sd = self.sd(mu) if sd_fixed is None else sd_fixed
        return (y - mu) / sd

    def deviance(self, y, mu) -> float:
        sd = self.sd(mu)
        return float(np.sum(((y - mu) / sd) ** 2) + np.sum(np.log(2 * np.pi * sd ** 2)))

    def describe(self) -> str:
        return f"Gaussian (sigma0 = {self.sigma0:g}, rel = {self.rel:g})"
