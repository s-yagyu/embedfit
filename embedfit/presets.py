"""Ready-made combinations of the parts for each technique.

A technique is not a subclass: it is a choice of signal shape, background
terms and noise model.  Techniques that share a noise model (counting PYS and
inverse photoemission) or a signal shape (counting and current PYS) reuse the
same parts.  Add a technique by writing a function like these.
"""
from __future__ import annotations

import warnings

from .model import Constant, ExpDecay, FluxProportional, ForwardModel, LinearDrift, PowerDecay, PowerThreshold
from .noise import Gaussian, Poisson

#: settings used in the JSA paper for the AC series (photon energy in eV)
PYS_FIT = dict(position_offsets=(-0.2, 0.0, 0.2))
PYS_VERDICT = dict(controls=("c", "n"), n_low=5)
PYS_DISPLAY = dict(n_low=5, k_sigma=3.0, run=2, gap=0.3)
PYS_JS_MARGIN = 0.35


def pys_counting(n: float = 2.0, free_n: bool = False) -> ForwardModel:
    """Counting PYS (open counter, AC series), JSA Eq. (13):
    lambda = d + c Phi + k Phi max(E - Eth, 0)^n,  mu = T_g G^{-1}(lambda),  N ~ Poisson(mu).
    c is fixed at 0 and n at its value; both are freed by the verdict's control models."""
    return ForwardModel(signal=PowerThreshold(n, fix_n=not free_n),
                        background=[Constant("d", lower=0.0), FluxProportional("c")],
                        noise=Poisson(), name="PYS counting")


def pys_current(sigma0: float, rel: float = 0.0, n: float = 2.0, background: str = "const",
                tau0: float = 60.0) -> ForwardModel:
    """Current-method PYS (electrometer) -- EXPERIMENTAL.

    I = D [+ g(t)] + c Phi + k Phi max(E - Eth, 0)^n with Gaussian noise
    sd = sqrt(sigma0^2 + (rel I)^2).  background: 'const' | 'exp' | 'power' | 'linear'
    (time-dependent terms need Spectrum.time, the time since the voltage step).

    Not validated on measured data: the relaxation models and the noise model
    have been checked on synthetic data only, and the electrometer's filter is
    not modelled.  sigma0 and rel must be measured separately (e.g. a dark sweep).
    """
    warnings.warn("embedfit.presets.pys_current is experimental: not validated on measured data",
                  stacklevel=2)
    bg = [Constant("d", lower=-float("inf")), FluxProportional("c")]
    if background == "exp":
        bg.append(ExpDecay(tau0=tau0))
    elif background == "power":
        bg.append(PowerDecay())
    elif background == "linear":
        bg.append(LinearDrift())
    elif background != "const":
        raise ValueError("background must be 'const', 'exp', 'power' or 'linear'")
    return ForwardModel(signal=PowerThreshold(n), background=bg, noise=Gaussian(sigma0, rel),
                        name=f"PYS current ({background})", experimental=True)
