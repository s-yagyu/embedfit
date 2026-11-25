"""Checks on synthetic data with known truth."""
import warnings

import numpy as np
import pytest

from embedfit import Analysis, fit, pys_current, recover_counts
from embedfit.conventional import path_c
from embedfit.synth import counting_pys, current_pys


def test_recovery_exact():
    rng = np.random.default_rng(1)
    spec, truth = counting_pys(delta=2.0, rng=rng)
    rec, info = recover_counts(spec)
    assert info.integer_ok
    assert info.gate_time == pytest.approx(4.0)
    # j_s is estimated from var = mean, so the recovered counts may be off by a common integer
    off = rec.counts() - truth["counts"]
    assert np.ptp(off) == 0
    assert abs(off[0]) < 25


def test_counting_unbiased_and_coverage():
    rng = np.random.default_rng(2)
    est, se, normal = [], [], 0
    for _ in range(40):
        spec, truth = counting_pys(delta=2.0, rng=rng)
        an = Analysis(spec).run()
        est.append(an.result.estimate)
        se.append(an.result.estimate_se)
        normal += an.verdict_.level == "normal"
    est, se = np.array(est), np.array(se)
    assert abs(est.mean() - 4.95) < 3 * est.std(ddof=1) / np.sqrt(len(est)) + 0.003
    assert np.mean(np.abs(est - 4.95) <= 1.96 * se) > 0.85
    assert normal >= 30


def test_decaying_background_flagged():
    rng = np.random.default_rng(3)
    flagged = 0
    for _ in range(20):
        spec, _ = counting_pys(delta=2.0, rng=rng, background=lambda x: 4.0 + 2.5 * np.exp(-(x - 3.8) / 0.5))
        an = Analysis(spec).run()
        flagged += an.verdict_.level != "normal"
    assert flagged >= 3


def test_display_flat():
    rng = np.random.default_rng(4)
    spec, _ = counting_pys(delta=2.0, rng=rng)
    an = Analysis(spec).run()
    for method in ("fit", "window"):
        dy = an.display(method=method)
        chk = dy.flat_check(spec.x <= 4.55)
        assert abs(chk["mean_z"]) < 1.5


def test_path_c_close_to_forward():
    rng = np.random.default_rng(5)
    spec, truth = counting_pys(delta=2.0, rng=rng, output="counts")
    r = path_c(spec)
    an = Analysis(spec).run()
    assert abs(r.Eth - an.result.estimate) < 0.02


def test_current_experimental():
    rng = np.random.default_rng(6)
    spec, truth = current_pys(relax="exp", amp=0.5, tau=60.0, tau_f=0.01, rng=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        const = pys_current(sigma0=0.01, rel=0.01)
        expo = pys_current(sigma0=0.01, rel=0.01, background="exp")
    r0 = fit(spec, const)
    r1 = fit(spec, expo)
    assert r1.deviance < r0.deviance - 10
    assert abs(r1.estimate - 5.0) < abs(r0.estimate - 5.0) + 0.01
    assert abs(r1.estimate - 5.0) < 0.05
