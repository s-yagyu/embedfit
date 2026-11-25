"""Figures (matplotlib, optional dependency).  Functions only draw; every
number comes from the analysis objects, and :mod:`embedfit.export` writes the
same numbers to CSV.

overview(an)   2 x 3 panels: (a) true rate and its parts, (b) reference Phi,
               (c) recorded rate with the fitted expectation (+-2 sd band),
               (d) corrected-value coordinates: instrument value and display
               yield with the model, (e) standardized residuals with the z_low
               window, (f) verdict text.
display(an)    display yield only (for papers); power=1/n draws Y^(1/n) with
               only the model curve raised to 1/n.
scans(ss)      estimates +- SE of repeated scans, marked by verdict.
"""
from __future__ import annotations

import numpy as np

from .checks import low_window
from .fit import evaluate

INK = "#222222"
MUTED = "#8a8a8a"
MODEL = "#1f5fbf"      # model curves
ACCENT = "#c2410c"     # instrument output (for contrast with the display yield)
BAND = "#1f5fbf"
LEVEL_MARK = {"normal": ("o", INK), "caution": ("s", ACCENT), "abnormal": ("x", MUTED)}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def overview(an, xlabel: str = "Photon energy (eV)", figsize=(11, 6.5)):
    plt = _plt()
    r, sp, v, dy = an.result, an.spectrum, an.verdict_, an.display_
    ev = evaluate(r, sp)
    x = sp.x
    counting = sp.is_counting
    T = sp.T()
    fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    (a, b, c), (d, e, f) = axs
    for ax in axs.flat[:5]:
        _style(ax)

    # (a) true rate
    a.plot(x, ev["lam"], color=MODEL, lw=2, label="λ (true rate)")
    a.plot(x, ev["background"], color=MUTED, lw=1.5, ls="--", label="background")
    a.plot(x, ev["signal_rate"], color=INK, lw=1.2, ls=":", label="k Φ f")
    a.set_title("(a) model, true rate", loc="left", fontsize=10)
    a.set_ylabel("rate (cps)" if counting else "current")
    a.legend(frameon=False, fontsize=8)

    # (b) reference
    b.plot(x, sp.reference, color=INK, lw=1.5)
    b.set_title("(b) reference quantity Φ", loc="left", fontsize=10)
    b.set_ylabel("Φ")

    # (c) recorded values and expectation
    obs = sp.counts() / T if counting else sp.signal
    mu_rate = ev["mu"] / T if counting else ev["mu"]
    sd = (np.sqrt(np.maximum(ev["mu"], 1e-9)) / T) if counting else r.model.noise.sd(ev["mu"])
    c.fill_between(x, mu_rate - 2 * sd, mu_rate + 2 * sd, color=BAND, alpha=0.15, lw=0, label="±2 sd")
    c.plot(x, obs, "o", ms=4, mfc="none", mec=INK, mew=1, label="recorded")
    c.plot(x, mu_rate, color=MODEL, lw=2, label="expectation")
    c.set_title("(c) recorded rate and fitted expectation" if counting else "(c) recorded current and fit",
                loc="left", fontsize=10)
    c.set_ylabel("N/T (cps)" if counting else "current")
    c.legend(frameon=False, fontsize=8)

    # (d) corrected-value coordinates
    if dy is not None:
        mod = dy.model if dy.model is not None else ev["kf"]
        d.fill_between(x, mod - 2 * dy.sigma, mod + 2 * dy.sigma, color=BAND, alpha=0.15, lw=0, label="±2 σ_y")
        if dy.instrument is not None:
            d.plot(x, dy.instrument, "o", ms=4, mfc="none", mec=ACCENT, mew=1, label="instrument (no clip)")
        d.plot(x, dy.y, "o", ms=4, color=INK, label=f"display yield (d̂ from {dy.method})")
        d.plot(x, mod, color=MODEL, lw=2, label="model c + k f")
        d.axhline(0, color=MUTED, lw=0.8)
        d.legend(frameon=False, fontsize=8)
    d.set_title("(d) corrected value", loc="left", fontsize=10)
    d.set_ylabel("yield")

    # (e) residuals
    xf = r.x
    w = low_window(r, an.settings["verdict"].get("n_low", 5), an.settings["verdict"].get("x_low"))
    e.axvspan(xf[w].min() - 1e-3, xf[w].max() + 1e-3, color=MUTED, alpha=0.15, lw=0, label="z_low window")
    e.axhline(0, color=MUTED, lw=0.8)
    for yv in (-2, 2):
        e.axhline(yv, color=MUTED, lw=0.6, ls="--")
    e.plot(xf, r.r, "o", ms=4, color=INK, label="r = (y − μ)/sd")
    e.set_title("(e) standardized residuals", loc="left", fontsize=10)
    e.set_ylabel("r")
    e.legend(frameon=False, fontsize=8)

    # (f) numbers
    f.axis("off")
    lines = [an.input.label or "spectrum", "",
             f"{r.model.position} = {r.estimate:.4f} ± {r.estimate_se:.4f}",
             f"χ²/dof = {r.chi2_red:.2f}  (limit {v.chi2_limit:.2f})" if v else f"χ²/dof = {r.chi2_red:.2f}"]
    if v is not None:
        lines += [f"z_low = {v.z_low:+.2f}", f"verdict: {v.level}"]
        if v.level == "caution":
            lines += [f"cause: {v.cause}", f"systematic shift {v.shift:+.4f}"]
    if an.recovery is not None:
        lines += ["", f"T_g = {an.recovery.gate_time:g} s, b_s = {an.recovery.dark_rate:.3f}",
                  f"integer check {'passed' if an.recovery.integer_ok else 'FAILED'}"]
    if r.model.experimental:
        lines += ["", "EXPERIMENTAL model"]
    f.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9, color=INK)

    for ax in (d, e):
        ax.set_xlabel(xlabel)
    c.set_xlabel(xlabel)
    return fig


def display(an, power=None, xlabel: str = "Photon energy (eV)", figsize=(5.5, 4), show_instrument: bool = False):
    """Display yield with the model and the ±2 σ band.  With power=1/n the axis is
    Y^(1/n): only the model curve is raised to 1/n, the data are not (their variance
    diverges near zero), so the data are drawn on a second figure panel in Y."""
    plt = _plt()
    dy = an.display_
    x = dy.x
    mod = dy.model
    if power is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _style(ax)
        ax.fill_between(x, mod - 2 * dy.sigma, mod + 2 * dy.sigma, color=BAND, alpha=0.15, lw=0, label="±2 σ_y")
        if show_instrument and dy.instrument is not None:
            ax.plot(x, dy.instrument, "o", ms=4, mfc="none", mec=ACCENT, label="instrument (no clip)")
        ax.plot(x, dy.y, "o", ms=4, color=INK, label="display yield")
        ax.plot(x, mod, color=MODEL, lw=2, label="model")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("yield")
        ax.legend(frameon=False, fontsize=8)
        return fig
    fig, (a, b) = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]), constrained_layout=True)
    for ax in (a, b):
        _style(ax)
    a.fill_between(x, mod - 2 * dy.sigma, mod + 2 * dy.sigma, color=BAND, alpha=0.15, lw=0, label="±2 σ_y")
    a.plot(x, dy.y, "o", ms=4, color=INK, label="display yield")
    a.plot(x, mod, color=MODEL, lw=2, label="model")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_ylabel("yield")
    a.legend(frameon=False, fontsize=8)
    b.plot(x, np.clip(mod, 0, None) ** power, color=MODEL, lw=2)
    b.set_ylabel(f"model$^{{{power:.3g}}}$")
    b.set_title("model only (data not transformed)", loc="left", fontsize=9)
    for ax in (a, b):
        ax.set_xlabel(xlabel)
    return fig


def scans(ss, ylabel: str = "estimate (eV)", figsize=(6, 3.8)):
    plt = _plt()
    rows = ss.table()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style(ax)
    seen = set()
    for row in rows:
        m, col = LEVEL_MARK[row["verdict"]]
        lab = row["verdict"] if row["verdict"] not in seen else None
        seen.add(row["verdict"])
        ax.errorbar(row["scan"], row["estimate"], yerr=row["se"], fmt=m, color=col, ms=6, capsize=2, lw=1,
                    mfc="none" if row["verdict"] == "caution" else col, label=lab)
    ax.set_xlabel("scan")
    ax.set_ylabel(ylabel)
    ax.set_xticks([row["scan"] for row in rows])
    ax.legend(frameon=False, fontsize=8, title="verdict", title_fontsize=8)
    return fig
