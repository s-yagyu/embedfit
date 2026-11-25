"""Forward model, JSA Eqs. (1)-(2):

    lambda(x) = background(x, t, Phi) + k Phi(x) f(x; theta)
    m_i       = G^{-1}(lambda_i)                (expected recorded rate)
    mu_i      = T_i m_i  (counts)   or   m_i   (current)

The model is assembled from parts that vary independently between techniques:

* a :class:`Signal` f(x; theta) (threshold power law, ...),
* a list of :class:`Background` terms (constant d, flux-proportional c Phi,
  decaying with time, ...),
* the detector correction G (taken from the spectrum),
* a noise model (see :mod:`embedfit.noise`).

Every quantity is a named :class:`Param`; any parameter can be fixed or freed,
which is how the control models of the verdict are built.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .noise import NoiseModel, Poisson


def relu(z):
    return np.clip(z, 0.0, None)


@dataclass
class Param:
    name: str
    value: float = 0.0
    lower: float = -np.inf
    upper: float = np.inf
    fixed: bool = False
    log: bool = False          # optimized as log(value); value must stay > 0
    role: str = ""             # "signal" | "background" | "scale"


@dataclass
class Context:
    """Arrays of the points being fitted."""

    x: np.ndarray
    phi: np.ndarray
    T: np.ndarray
    t: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# signal shapes f(x; theta)
# --------------------------------------------------------------------------- #

class Signal:
    """Physical response f(x; theta), multiplied by k Phi in the model."""

    position: Optional[str] = None     # name of the main position parameter (multi-start)

    def params(self) -> list[Param]:
        raise NotImplementedError

    def __call__(self, p: dict, ctx: Context) -> np.ndarray:
        raise NotImplementedError


class PowerThreshold(Signal):
    """f = max(x - Eth, 0)^n (JSA Eq. (12)); n = 2 for metals (Fowler), 3 is
    customary for organic semiconductors."""

    position = "Eth"

    def __init__(self, n: float = 2.0, fix_n: bool = True, n_bounds=(0.5, 4.0)):
        self.n, self.fix_n, self.n_bounds = n, fix_n, n_bounds

    def params(self) -> list[Param]:
        return [Param("Eth", np.nan, role="signal"),
                Param("n", self.n, *self.n_bounds, fixed=self.fix_n, role="signal")]

    def __call__(self, p, ctx):
        return relu(ctx.x - p["Eth"]) ** p["n"]


# --------------------------------------------------------------------------- #
# background terms
# --------------------------------------------------------------------------- #

class Background:
    """A term added to the true rate lambda."""

    flux_proportional = False   # True for terms that follow Phi (not subtracted in the display)

    def params(self) -> list[Param]:
        raise NotImplementedError

    def __call__(self, p: dict, ctx: Context) -> np.ndarray:
        raise NotImplementedError

    def guess(self, ctx: Context, level: float) -> dict:
        return {}


class Constant(Background):
    """Background independent of the reference quantity: d (dark count, offset)."""

    def __init__(self, name: str = "d", lower: float = 0.0):
        self.name, self.lower = name, lower

    def params(self):
        return [Param(self.name, 0.0, self.lower, np.inf, role="background")]

    def __call__(self, p, ctx):
        return np.full_like(ctx.x, p[self.name])

    def guess(self, ctx, level):
        return {self.name: max(level, 0.1) if self.lower >= 0 else level}


class FluxProportional(Background):
    """Background proportional to the reference quantity: c Phi (stray light).
    Fixed at 0 by default; freed as the background control model."""

    flux_proportional = True

    def __init__(self, name: str = "c", fixed: bool = True, value: float = 0.0):
        self.name, self.fixed, self.value = name, fixed, value

    def params(self):
        return [Param(self.name, self.value, fixed=self.fixed, role="background")]

    def __call__(self, p, ctx):
        return p[self.name] * ctx.phi


class ExpDecay(Background):
    """A exp(-(t - t0)/tau): relaxation after a voltage step (current method).
    Needs the time of each reading."""

    def __init__(self, amp: str = "A", tau: str = "tau_bg", tau0: float = 60.0, fix_tau: bool = False):
        self.amp, self.tau, self.tau0, self.fix_tau = amp, tau, tau0, fix_tau

    def params(self):
        return [Param(self.amp, 0.0, role="background"),
                Param(self.tau, self.tau0, 1e-3, np.inf, fixed=self.fix_tau, log=True, role="background")]

    def __call__(self, p, ctx):
        if ctx.t is None:
            raise ValueError("ExpDecay needs the time of each reading (Spectrum.time)")
        return p[self.amp] * np.exp(-(ctx.t - ctx.t.min()) / p[self.tau])


class PowerDecay(Background):
    """A (t / t0)^(-m): Curie-von Schweidler relaxation.  t is the time since the
    voltage step, so it must not start at 0."""

    def __init__(self, amp: str = "A", m: str = "m_bg", m0: float = 0.8):
        self.amp, self.m, self.m0 = amp, m, m0

    def params(self):
        return [Param(self.amp, 0.0, role="background"),
                Param(self.m, self.m0, 0.05, 3.0, role="background")]

    def __call__(self, p, ctx):
        if ctx.t is None or ctx.t.min() <= 0:
            raise ValueError("PowerDecay needs positive times since the voltage step")
        return p[self.amp] * (ctx.t / ctx.t.min()) ** (-p[self.m])


class LinearDrift(Background):
    """s (t - t0): linear drift during the sweep."""

    def __init__(self, name: str = "s"):
        self.name = name

    def params(self):
        return [Param(self.name, 0.0, role="background")]

    def __call__(self, p, ctx):
        if ctx.t is None:
            raise ValueError("LinearDrift needs the time of each reading (Spectrum.time)")
        return p[self.name] * (ctx.t - ctx.t.min())


# --------------------------------------------------------------------------- #
# forward model
# --------------------------------------------------------------------------- #

@dataclass
class ForwardModel:
    """lambda = sum(background) + k Phi f;  expected record = T G^{-1}(lambda)
    (counts) or G^{-1}(lambda) (current)."""

    signal: Signal
    background: list = field(default_factory=lambda: [Constant()])
    noise: NoiseModel = field(default_factory=Poisson)
    scale: str = "k"
    name: str = ""
    experimental: bool = False

    def params(self) -> list[Param]:
        ps = [Param(self.scale, 1.0, 0.0, np.inf, log=True, role="scale")] + self.signal.params()
        for b in self.background:
            ps += b.params()
        names = [p.name for p in ps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        return ps

    @property
    def position(self) -> Optional[str]:
        return self.signal.position

    # -- evaluation ------------------------------------------------------- #
    def background_rate(self, p: dict, ctx: Context, flux: Optional[bool] = None) -> np.ndarray:
        """Sum of the background terms; flux=False keeps only the terms that do
        not follow Phi (what the display yield subtracts)."""
        out = np.zeros_like(ctx.x)
        for b in self.background:
            if flux is None or b.flux_proportional == flux:
                out = out + b(p, ctx)
        return out

    def signal_rate(self, p: dict, ctx: Context) -> np.ndarray:
        return p[self.scale] * ctx.phi * self.signal(p, ctx)

    def rate(self, p: dict, ctx: Context) -> np.ndarray:
        """True rate lambda (before the detector losses)."""
        return self.background_rate(p, ctx) + self.signal_rate(p, ctx)

    def expected(self, p: dict, ctx: Context, detector) -> np.ndarray:
        """Expected recorded value mu (counts, or the current)."""
        m = detector.G_inv(self.rate(p, ctx))
        return ctx.T * m if self.noise.counting else m

    # -- variants --------------------------------------------------------- #
    def copy(self) -> "ForwardModel":
        return copy.deepcopy(self)

    def describe(self) -> str:
        bg = " + ".join(type(b).__name__ for b in self.background)
        return f"{self.name or 'model'}: lambda = [{bg}] + {self.scale} Phi {type(self.signal).__name__}; noise {self.noise.describe()}"
