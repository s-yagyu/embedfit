"""One object that runs the whole analysis of a spectrum::

    an = Analysis.from_file("Au_1.dat")          # Riken AC reader by default
    an.run()                                     # recover counts, fit, verdict, display yield
    print(an.report_text())
    an.plot()                                    # overview figure
    an.to_csv("out/")                            # the values behind the figures

and :class:`ScanSet` for repeated scans.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from . import presets
from .checks import Verdict, dispersion, verdict
from .data import Spectrum
from .display import DisplayYield, display_yield
from .fit import FitResult, fit
from .model import ForwardModel
from .recover import RecoveryInfo, recover_counts


def _settings(model: ForwardModel) -> dict:
    if model.name.startswith("PYS"):
        return dict(fit=dict(presets.PYS_FIT), verdict=dict(presets.PYS_VERDICT), display=dict(presets.PYS_DISPLAY),
                    margin=presets.PYS_JS_MARGIN)
    return dict(fit={}, verdict={}, display={}, margin=0.35)


class Analysis:
    """Model-embedded analysis of one spectrum.

    Parameters
    ----------
    spectrum
        input as read from the instrument.
    model
        forward model; default: counting PYS (n = 2) for counting data.
    xrange
        fitting range Omega (None: all points, as in the JSA paper).
    fixed
        parameters to fix in the base fit (e.g. {"n": 3}).
    """

    def __init__(self, spectrum: Spectrum, model: Optional[ForwardModel] = None, xrange=None,
                 fixed: Optional[dict] = None, settings: Optional[dict] = None):
        if model is None:
            if not spectrum.is_counting:
                raise ValueError("give a model for non-counting data (e.g. presets.pys_current)")
            model = presets.pys_counting()
        self.input = spectrum
        self.model = model
        self.xrange = xrange
        self.fixed = dict(fixed or {})
        self.settings = _settings(model) if settings is None else settings
        self.spectrum: Optional[Spectrum] = None
        self.recovery: Optional[RecoveryInfo] = None
        self.result: Optional[FitResult] = None
        self.verdict_: Optional[Verdict] = None
        self.display_: Optional[DisplayYield] = None

    @classmethod
    def from_file(cls, path: str, reader: str = "riken_ac", **kw) -> "Analysis":
        from .readers import read
        return cls(read(reader, path), **kw)

    # -- steps ---------------------------------------------------------- #
    def prepare(self) -> Spectrum:
        """Path B: recover counts when the output holds only a dark-subtracted rate."""
        if self.input.signal_kind == "rate_dark_subtracted":
            self.spectrum, self.recovery = recover_counts(
                self.input, self.model, xrange=self.xrange, margin=self.settings["margin"],
                fit_kwargs=dict(self.settings["fit"], fixed=self.fixed) if self.fixed else self.settings["fit"])
        else:
            self.spectrum = self.input
        return self.spectrum

    def fit(self) -> FitResult:
        if self.spectrum is None:
            self.prepare()
        self.result = fit(self.spectrum, self.model, xrange=self.xrange, fixed=self.fixed or None, **self.settings["fit"])
        return self.result

    def verdict(self, **kw) -> Verdict:
        if self.result is None:
            self.fit()
        opts = dict(self.settings["verdict"])
        opts.update(kw)
        opts.setdefault("fit_kwargs", self.settings["fit"])
        self.verdict_ = verdict(self.result, xrange=self.xrange, **opts)
        return self.verdict_

    def display(self, method="fit", **kw) -> DisplayYield:
        if self.result is None:
            self.fit()
        opts = dict(self.settings["display"])
        opts.update(kw)
        self.display_ = display_yield(self.spectrum, self.result, method=method, **opts)
        return self.display_

    def run(self) -> "Analysis":
        self.prepare()
        self.fit()
        self.verdict()
        self.display()
        return self

    # -- output --------------------------------------------------------- #
    def report(self) -> dict:
        """Values to report.  The estimate is withheld (None) when the verdict is abnormal."""
        r, v = self.result, self.verdict_
        out = dict(label=self.input.label, model=self.model.describe(), points=int(r.mask.sum()),
                   chi2_red=r.chi2_red, dof=r.dof, params=dict(r.params), se=dict(r.se))
        if self.recovery is not None:
            out["recovery"] = dict(gate_time=self.recovery.gate_time, dark_rate=self.recovery.dark_rate,
                                   j_s=self.recovery.j_s, integer_ok=self.recovery.integer_ok,
                                   max_misfit=float(self.recovery.misfit.max()))
        if v is not None:
            out.update(verdict=v.level, cause=v.cause, z_low=v.z_low, chi2_limit=v.chi2_limit,
                       estimate=None if v.level == "abnormal" else v.estimate,
                       se_estimate=None if v.level == "abnormal" else v.se,
                       systematic_shift=v.shift if v.level == "caution" else 0.0,
                       controls={k: dict(delta_deviance=c.delta_deviance, shift=c.shift, value=c.value)
                                 for k, c in v.controls.items()})
        return out

    def report_text(self) -> str:
        lines = [f"[{self.input.label}]"]
        if self.recovery is not None:
            lines.append("recovery: " + self.recovery.summary())
        lines.append(self.result.summary())
        if self.verdict_ is not None:
            lines.append("verdict: " + self.verdict_.summary())
        if self.display_ is not None:
            lines.append(f"display yield: d_hat ({self.display_.method}) = {self.display_.d_hat.mean():.4g} "
                         f"+- {self.display_.d_se:.3g}")
        return "\n".join(lines)

    def plot(self, kind: str = "overview", **kw):
        from . import plot
        if kind == "overview":
            return plot.overview(self, **kw)
        if kind == "display":
            return plot.display(self, **kw)
        raise ValueError("kind must be 'overview' or 'display'")

    def to_csv(self, folder: str, stem: Optional[str] = None) -> list:
        from .export import analysis_to_csv
        return analysis_to_csv(self, folder, stem)


class ScanSet:
    """Repeated scans of one sample, each analysed on its own."""

    def __init__(self, spectra: Sequence[Spectrum], **kw):
        self.analyses = [Analysis(s, **kw) for s in spectra]

    @classmethod
    def from_folder(cls, folder: str, **kw) -> "ScanSet":
        from .readers import read_ac_folder
        return cls(read_ac_folder(folder), **kw)

    def run(self) -> "ScanSet":
        for a in self.analyses:
            a.run()
        return self

    def table(self) -> list:
        """One row per scan.  As in ``report()``, the estimate and its SE are withheld (nan)
        when the verdict is abnormal."""
        rows = []
        for i, a in enumerate(self.analyses, 1):
            v, r = a.verdict_, a.result
            withhold = v is not None and v.level == "abnormal"
            rows.append(dict(scan=i, label=a.input.label, estimate=np.nan if withhold else r.estimate,
                             se=np.nan if withhold else r.estimate_se, chi2_red=r.chi2_red,
                             z_low=v.z_low, verdict=v.level, cause=v.cause, shift=v.shift,
                             dD_c=v.controls["c"].delta_deviance if "c" in v.controls else np.nan,
                             dD_n=v.controls["n"].delta_deviance if "n" in v.controls else np.nan))
        return rows

    def dispersion(self, mask=None):
        """Noise-model check over the scans (variance against mean of the counts).
        Meaningful only for repeated scans of a stable sample."""
        return dispersion(np.array([a.spectrum.counts() for a in self.analyses]), mask)

    def plot(self, **kw):
        from . import plot
        return plot.scans(self, **kw)

    def to_csv(self, path: str) -> str:
        from .export import scans_to_csv
        return scans_to_csv(self, path)
