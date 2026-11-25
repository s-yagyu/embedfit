"""Checks of a fit and of the noise model.

Verdict (JSA section 3.5, Table 3)
    normal   : chi2/dof within the upper 1 % point of chi2(dof)/dof and |z_low| <= 2
    caution  : an indicator fails and a control model (one added parameter) lowers
               the deviance by more than 3.84; its change of the estimate is
               reported as a systematic uncertainty.  For a flux-proportional
               background c the sign refines the cause.
    abnormal : an indicator fails and no control model explains it, or the fit
               did not converge; no estimate is reported.

Noise-model checks that need no model (once per instrument, not per spectrum)
    integer check of recovered counts (see :mod:`embedfit.recover`) and
    :func:`dispersion` (variance against mean over repeated scans).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .fit import FitResult, fit

CHI_ALPHA = 0.01     # upper-tail probability of the chi2 indicator
Z_CRIT = 2.0         # |z_low| criterion
DD_CRIT = 3.84       # upper 5 % point of chi2(1)


def chi2_limit(dof: int, alpha: float = CHI_ALPHA) -> float:
    """Upper alpha point of chi2/dof."""
    return float(chi2_dist.ppf(1 - alpha, dof) / dof)


def low_window(result: FitResult, n_low: Optional[int] = 5, x_low: Optional[float] = None) -> np.ndarray:
    """Points of the z_low window among the fitted points: x <= x_low, or the
    n_low lowest points.  Choose it before looking at the residuals."""
    x = result.x
    if x_low is not None:
        return x <= x_low + 1e-9
    w = np.zeros(len(x), bool)
    w[np.argsort(x)[:n_low]] = True
    return w


def z_low(result: FitResult, n_low: Optional[int] = 5, x_low: Optional[float] = None) -> float:
    """z_low = sum of the standardized residuals in the window / sqrt(M), JSA Eq. (8)."""
    w = low_window(result, n_low, x_low)
    return float(result.r[w].sum() / np.sqrt(w.sum()))


@dataclass
class Control:
    name: str            # parameter freed
    role: str            # 'background' | 'response'
    delta_deviance: float
    shift: float         # change of the estimate (control - base)
    value: float         # fitted value of the freed parameter
    result: FitResult = field(repr=False, default=None)


@dataclass
class Verdict:
    level: str                       # 'normal' | 'caution' | 'abnormal'
    cause: str
    shift: float                     # systematic change of the estimate (caution only)
    estimate: float
    se: float
    chi2_red: float
    chi2_limit: float
    z_low: float
    chi2_ok: bool
    z_ok: bool
    controls: dict = field(default_factory=dict)

    @property
    def reported(self) -> Optional[tuple]:
        """(estimate, se, systematic shift) or None when abnormal."""
        if self.level == "abnormal":
            return None
        return self.estimate, self.se, self.shift

    def summary(self, unit: str = "eV", scale: float = 1.0) -> str:
        s = (f"{self.level}: estimate {self.estimate:.4f} {unit} +- {self.se * scale:.4g}; chi2/dof {self.chi2_red:.2f} "
             f"(limit {self.chi2_limit:.2f}), z_low {self.z_low:+.2f}")
        if self.controls:
            s += "; controls " + ", ".join(f"{c.name}: dD {c.delta_deviance:.2f}, shift {c.shift:+.4g}"
                                         for c in self.controls.values())
        if self.level == "caution":
            s += f" -> cause {self.cause}, systematic shift {self.shift:+.4g}"
        return s


def background_subcause(c: float) -> str:
    """c < 0: background falling toward higher flux (falling during the sweep when Phi
    rises with x); c > 0: flux-proportional background (stray light)."""
    return "background (falling)" if c < 0 else "background (flux-proportional)"


def control_fit(base: FitResult, name: str, xrange=None, **kw) -> Control:
    """Refit with one more free parameter and compare deviances."""
    model = base.model
    params = {p.name: p for p in model.params()}
    fixed = {k: base.params[k] for k in params if k not in base.free and k != name}
    start = {k: base.params[k] for k in base.free}
    res = fit(base.spectrum, model, xrange=xrange, fixed=fixed, free=[name], start=start,
              position0=base.estimate, **kw)
    role = "background" if params[name].role == "background" else "response"
    return Control(name, role, base.deviance - res.deviance, res.estimate - base.estimate, res.params[name], res)


def verdict(base: FitResult, controls: Sequence[str] = ("c", "n"), xrange=None, n_low: Optional[int] = 5,
            x_low: Optional[float] = None, always_controls: bool = False, alpha: float = CHI_ALPHA,
            z_crit: float = Z_CRIT, dd_crit: float = DD_CRIT, fit_kwargs: Optional[dict] = None) -> Verdict:
    """Three-level verdict of a fit.  `controls` are parameters of the model that the
    base fit keeps fixed (PYS: c for the background shape, n for the response shape).
    Pass the same `xrange` as the base fit."""
    lim = chi2_limit(base.dof, alpha)
    chi_ok = bool(chi2_dist.sf(base.chi2, base.dof) >= alpha)
    z = z_low(base, n_low, x_low)
    z_ok = bool(abs(z) <= z_crit)
    v = Verdict("normal", "", 0.0, base.estimate, base.estimate_se, base.chi2_red, lim, z, chi_ok, z_ok)
    if not base.success:
        v.level, v.cause = "abnormal", "fit did not converge"
        return v
    if chi_ok and z_ok and not always_controls:
        return v
    for name in controls:
        if name in base.free:
            continue
        v.controls[name] = control_fit(base, name, xrange=xrange, **(fit_kwargs or {}))
    if chi_ok and z_ok:
        return v
    best = None
    for c in v.controls.values():
        if c.delta_deviance > dd_crit and (best is None or c.delta_deviance > best.delta_deviance):
            best = c
    if best is None:
        v.level = "abnormal"
        return v
    v.level, v.shift = "caution", best.shift
    if best.role == "background":
        flux = any(getattr(b, "flux_proportional", False) and getattr(b, "name", None) == best.name
                   for b in base.model.background)
        v.cause = background_subcause(best.value) if flux else "background shape"
    else:
        v.cause = "response shape"
    return v


# --------------------------------------------------------------------------- #
# noise-model checks without a model
# --------------------------------------------------------------------------- #

@dataclass
class Dispersion:
    mean: np.ndarray          # per point, over scans
    var: np.ndarray
    ratio: np.ndarray         # var/mean per point
    pooled_ratio: float       # sum((N - mean)^2 / mean) / ((S - 1) P)
    p_value: float            # two-sided, chi2 with (S - 1) P dof

    def summary(self) -> str:
        return (f"index of dispersion pooled {self.pooled_ratio:.3f} (Poisson: 1), two-sided p = {self.p_value:.3f}; "
                f"per point median {np.median(self.ratio):.2f}")


def dispersion(counts, mask=None) -> Dispersion:
    """Variance against mean of the counts at each point over repeated scans
    (rows = scans).  For Poisson counts with a stable source the ratio is 1.
    Drift between scans raises it, dead time lowers it."""
    N = np.asarray(counts, float)
    if mask is not None:
        N = N[:, np.asarray(mask, bool)]
    S, P = N.shape
    m = N.mean(axis=0)
    v = N.var(axis=0, ddof=1)
    stat = float(np.sum((N - m) ** 2 / np.maximum(m, 1e-9)))
    dof = (S - 1) * P
    p = 2 * min(chi2_dist.sf(stat, dof), chi2_dist.cdf(stat, dof))
    return Dispersion(m, v, v / np.maximum(m, 1e-9), stat / dof, float(p))
