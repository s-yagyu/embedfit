"""Maximum-likelihood fit of a forward model to the recorded values.

The objective is the noise model's likelihood written as a sum of squared
residuals (Poisson: deviance residuals, JSA appendix B.3; Gaussian: residuals
divided by the sd at the model, iterated).  Standard errors come from the
Fisher information I = J^T W J, J = d mu / d theta, W = 1/var(mu).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares

from .data import Spectrum
from .model import Context, ForwardModel, Param

_TINY = 1e-9


@dataclass
class FitResult:
    model: ForwardModel
    spectrum: Spectrum          # the full spectrum that was passed in
    mask: np.ndarray            # points used (fitting range Omega)
    params: dict                # all parameters (fixed ones included)
    se: dict                    # standard errors of the free parameters (0 for fixed)
    free: list
    cov: np.ndarray
    obs: np.ndarray             # recorded values on the fitted points
    mu: np.ndarray              # expected recorded values
    lam: np.ndarray             # true rate lambda
    r: np.ndarray               # standardized residuals (y - mu)/sd(mu)
    chi2: float                 # Pearson chi2
    dof: int
    deviance: float
    success: bool
    ctx: Context = field(repr=False, default=None)

    def __getitem__(self, name):
        return self.params[name]

    @property
    def x(self) -> np.ndarray:
        return self.ctx.x

    @property
    def estimate(self) -> float:
        """Value of the model's position parameter (Eth for PYS)."""
        return self.params[self.model.position]

    @property
    def estimate_se(self) -> float:
        return self.se[self.model.position]

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof

    def rate_rec(self) -> np.ndarray:
        """Expected recorded rate mu/T (counts) or mu (current)."""
        return self.mu / self.ctx.T if self.model.noise.counting else self.mu

    def summary(self) -> str:
        lines = [self.model.describe(),
                 f"points {int(self.mask.sum())}, x {self.x.min():g}..{self.x.max():g}, dof {self.dof}, "
                 f"chi2/dof {self.chi2_red:.3f}, deviance {self.deviance:.2f}, converged {self.success}"]
        for k, v in self.params.items():
            if k in self.free:
                lines.append(f"  {k:>8} = {v:.6g} +- {self.se[k]:.3g}")
            else:
                lines.append(f"  {k:>8} = {v:.6g} (fixed)")
        return "\n".join(lines)


def _context(spec: Spectrum, mask) -> Context:
    return Context(x=spec.x[mask], phi=spec.reference[mask], T=spec.T()[mask],
                   t=None if spec.time is None else spec.time[mask])


def observed(spec: Spectrum, model: ForwardModel) -> np.ndarray:
    if model.noise.counting:
        return spec.counts()
    if spec.signal_kind != "current":
        raise ValueError("a Gaussian (current) model needs signal_kind='current'")
    return spec.signal


def range_mask(x, xrange) -> np.ndarray:
    if xrange is None:
        return np.ones(len(x), bool)
    lo, hi = xrange
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    return (x >= lo - 1e-9) & (x <= hi + 1e-9)


def onset(x, rate, sd, n_low: int = 5, k_sigma: float = 3.0, run: int = 2, gap: Optional[float] = None):
    """Rough onset of the signal (JSA appendix B.4): starting from the lowest
    n_low points, the first x above the window where `run` points in a row
    exceed mean + k_sigma sd; the window is extended to `gap` below it and the
    step repeated until it no longer changes.  Returns (onset, window mask).
    Used for starting values and for the signal-free window of the display."""
    order = np.argsort(x)
    xs, rs = x[order], rate[order]
    sds = np.broadcast_to(sd, x.shape)[order] if np.ndim(sd) else np.full(len(x), float(sd))
    gap = 0.125 * (xs[-1] - xs[0]) if gap is None else gap
    win = np.arange(len(xs)) < n_low
    on = xs[-1]
    for _ in range(50):
        level = rs[win].mean()
        s = np.sqrt(np.mean(sds[win] ** 2))
        above = rs > level + k_sigma * s
        on = xs[-1]
        for i in range(int(win.sum()), len(xs) - run + 1):
            if np.all(above[i:i + run]):
                on = xs[i]
                break
        new = xs <= on - gap + 1e-9
        if new.sum() < n_low:
            new = np.arange(len(xs)) < n_low
        if np.array_equal(new, win):
            break
        win = new
    mask = np.zeros(len(x), bool)
    mask[order[win]] = True
    return float(on), mask


def fit(spec: Spectrum, model: ForwardModel, xrange=None, fixed: Optional[dict] = None,
        free: Iterable[str] = (), start: Optional[dict] = None, position0: Optional[float] = None,
        position_offsets: Optional[Iterable[float]] = None, max_nfev: int = 5000, n_irls: int = 3) -> FitResult:
    """Fit `model` to `spec` by maximum likelihood.

    Parameters
    ----------
    xrange
        (lo, hi) of the fitting range Omega; None = all points.
    fixed
        {name: value} parameters to fix (e.g. ``{"n": 2, "c": 0}``).
    free
        names of parameters to free that the model fixes by default (control models).
    start
        {name: value} starting values that override the automatic guesses.
    position0, position_offsets
        starting value of the position parameter (Eth) and the offsets of the
        multi-start; the solution with the smallest objective is kept.
        Default: onset of the data and +-8 % of the range.  ``PYS_FIT`` uses the
        onset +-0.2 eV; the JSA paper used fixed starts 4.8, 5.0, 5.2 eV for Au,
        which give the same solutions on the Au data.
    """
    mask = range_mask(spec.x, xrange)
    ctx = _context(spec, mask)
    y = observed(spec, model)[mask]
    det = spec.detector
    noise = model.noise

    params = {p.name: p for p in model.params()}
    for k, v in (fixed or {}).items():
        params[k].value, params[k].fixed = float(v), True
    for k in free:
        params[k].fixed = False

    # -- starting values ---------------------------------------------------- #
    rate = y / ctx.T if noise.counting else y
    sd_rate = noise.sd(np.maximum(y, 1.0 if noise.counting else 0.0)) / (ctx.T if noise.counting else 1.0)
    on, low = onset(ctx.x, rate, sd_rate)
    level = float(det.G(np.array(rate[low].mean())))
    guess = {}
    for b in model.background:
        guess.update(b.guess(ctx, level))
    pos = model.position
    x0 = position0 if position0 is not None else on
    if position_offsets is None:
        span = ctx.x.max() - ctx.x.min()
        position_offsets = (-0.08 * span, 0.0, 0.08 * span)
    if pos is not None and np.isnan(params[pos].value):
        params[pos].lower = max(params[pos].lower, ctx.x.min())
        params[pos].upper = min(params[pos].upper, ctx.x.max())
    for k, v in guess.items():
        if not params[k].fixed:
            params[k].value = v
    for k, v in (start or {}).items():
        params[k].value = float(v)
    if pos is not None and not params[pos].fixed and np.isnan(params[pos].value):
        params[pos].value = x0
    # amplitude from the highest point
    if not params[model.scale].fixed and not (start and model.scale in start):
        trial = {k: p.value for k, p in params.items()}
        trial[model.scale] = 1.0
        if pos is not None:
            trial[pos] = x0
        top = model.signal_rate(trial, ctx)
        i = int(np.argmax(ctx.x))
        bg_top = model.background_rate(trial, ctx)[i]
        hi_rate = float(det.G(np.array(rate.max())))
        floor = 1.0 if noise.counting else 1e-6 * max(abs(hi_rate), 1e-12)
        params[model.scale].value = max(hi_rate - bg_top, floor) / max(top[i], 1e-9)

    names = [k for k, p in params.items() if not p.fixed]
    if not names:
        raise ValueError("no free parameter")

    def to_u(vals):
        return np.array([np.log(max(vals[k], _TINY)) if params[k].log else vals[k] for k in names], float)

    def unpack(u) -> dict:
        d = {k: p.value for k, p in params.items()}
        for k, v in zip(names, u):
            d[k] = float(np.exp(v)) if params[k].log else float(v)
        return d

    lb = np.array([np.log(max(params[k].lower, _TINY)) if params[k].log and params[k].lower > 0
                   else (-np.inf if params[k].log else params[k].lower) for k in names])
    ub = np.array([np.log(params[k].upper) if params[k].log and np.isfinite(params[k].upper)
                   else (np.inf if params[k].log else params[k].upper) for k in names])

    def mu_of(d):
        return model.expected(d, ctx, det)

    starts = []
    base = {k: p.value for k, p in params.items()}
    if pos is not None and pos in names:
        for off in position_offsets:
            s = dict(base)
            s[pos] = float(np.clip(x0 + off, params[pos].lower, params[pos].upper))
            starts.append(s)
    else:
        starts.append(base)

    best = None
    for s in starts:
        flb, fub = np.where(np.isfinite(lb), lb, 0.0), np.where(np.isfinite(ub), ub, 0.0)
        lo_in = np.where(np.isfinite(lb), flb + 1e-10 * np.maximum(1, np.abs(flb)), -np.inf)
        hi_in = np.where(np.isfinite(ub), fub - 1e-10 * np.maximum(1, np.abs(fub)), np.inf)
        u0 = np.clip(to_u(s), lo_in, hi_in)
        if noise.counting:
            res = least_squares(lambda u: noise.fit_residuals(y, mu_of(unpack(u))), u0, bounds=(lb, ub),
                                x_scale="jac", max_nfev=max_nfev)
            obj = 2 * res.cost
        else:
            sdf = noise.sd(np.maximum(y, 0.0))            # first pass at the data, then at the model
            u = u0
            for _ in range(n_irls):
                res = least_squares(lambda v: noise.fit_residuals(y, mu_of(unpack(v)), sdf), u, bounds=(lb, ub),
                                    x_scale="jac", max_nfev=max_nfev)
                u = res.x
                sdf = noise.sd(np.maximum(mu_of(unpack(u)), 0.0))
            obj = noise.deviance(y, mu_of(unpack(u)))
        if best is None or obj < best[0]:
            best = (obj, res)
    res = best[1]
    vals = unpack(res.x)
    mu = mu_of(vals)

    # -- Fisher information in the natural parameters ----------------------- #
    J = np.empty((len(y), len(names)))
    for a, k in enumerate(names):
        h = 1e-6 * max(abs(vals[k]), 1.0)
        vp, vm = dict(vals), dict(vals)
        vp[k] += h
        vm[k] -= h
        J[:, a] = (mu_of(vp) - mu_of(vm)) / (2 * h)
    F = J.T @ (J * noise.fisher_weights(np.maximum(mu, _TINY))[:, None])
    cov = np.linalg.pinv(F)
    se = {k: 0.0 for k in params}
    for a, k in enumerate(names):
        se[k] = float(np.sqrt(max(cov[a, a], 0.0)))
    r = noise.pearson(y, mu)
    return FitResult(model=model, spectrum=spec, mask=mask, params=vals, se=se, free=names, cov=cov, obs=y, mu=mu,
                     lam=model.rate(vals, ctx), r=r, chi2=float(np.sum(r ** 2)), dof=len(y) - len(names),
                     deviance=noise.deviance(y, mu), success=bool(res.success), ctx=ctx)


def evaluate(result: FitResult, spec: Optional[Spectrum] = None) -> dict:
    """Model quantities at every point of `spec` (default: the fitted spectrum,
    all points, also outside the fitting range)."""
    spec = result.spectrum if spec is None else spec
    ctx = _context(spec, np.ones(spec.n_points, bool))
    p, m = result.params, result.model
    lam = m.rate(p, ctx)
    return dict(x=ctx.x, lam=lam, background=m.background_rate(p, ctx), background_const=m.background_rate(p, ctx, flux=False),
                signal_rate=m.signal_rate(p, ctx), kf=m.signal_rate(p, ctx) / ctx.phi,
                mu=m.expected(p, ctx, spec.detector), phi=ctx.phi, T=ctx.T)
