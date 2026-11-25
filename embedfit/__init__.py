"""embedfit -- analysis of spectra normalized by a reference quantity, with the
instrument's corrections embedded in the model.

Instead of correcting the data (subtract the dark level, apply the detector
correction G, divide by the reference Phi) and fitting the result, the forward
model carries the corrections and is compared with the recorded counts:

    lambda = d + c Phi + k Phi f(x; theta),   mu = T G^{-1}(lambda),   N ~ Poisson(mu)

Reference: "Analysis of counting data normalized by a
reference quantity: embedding the instrument corrections in the model, with
application to photoelectron yield spectroscopy" (submitted).
"""
from .analysis import Analysis, ScanSet
from .checks import dispersion, verdict, z_low
from .data import Spectrum
from .detector import DetectorCorrection, Identity, NonParalyzable, RikenAC
from .display import display_yield
from .fit import evaluate, fit
from .model import (Constant, ExpDecay, FluxProportional, ForwardModel, LinearDrift, Param, PowerDecay,
                    PowerThreshold)
from .noise import Gaussian, Poisson
from .presets import pys_counting, pys_current
from .recover import recover_counts

__version__ = "0.1.0"

__all__ = ["Analysis", "ScanSet", "Spectrum", "fit", "evaluate", "verdict", "z_low", "dispersion", "display_yield",
           "recover_counts", "pys_counting", "pys_current", "ForwardModel", "PowerThreshold", "Constant",
           "FluxProportional", "ExpDecay", "PowerDecay", "LinearDrift", "Param", "Poisson", "Gaussian",
           "DetectorCorrection", "Identity", "NonParalyzable", "RikenAC"]
