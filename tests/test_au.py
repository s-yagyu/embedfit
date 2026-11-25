"""Reproduces the AC-5 Au results of the JSA paper (Fig. 4 for scan 1, Fig. 5, and the verdicts of Sec. 5.2),
reference values from the research code used for the paper."""
import os

import numpy as np
import pytest

from embedfit import ScanSet

DATA = os.environ.get("EMBEDFIT_AU_DIR",
                      os.path.join(os.path.dirname(__file__), "..", "examples", "data", "Au_AC5_20240229"))

# scan: (Eth eV, SE meV, chi2/dof, z_low, verdict, shift meV)
REF = {1: (4.9634, 13.0, 1.31, 3.19, "caution", -25.6), 2: (4.9715, 13.1, 0.74, 1.61, "normal", 0.0),
       3: (4.9484, 12.4, 1.13, 2.33, "caution", -24.5), 4: (4.9429, 13.0, 0.88, 0.93, "normal", 0.0),
       5: (4.9376, 13.7, 0.97, 1.55, "normal", 0.0), 6: (4.9438, 14.2, 0.77, 1.43, "normal", 0.0),
       7: (4.9489, 11.7, 0.93, 3.51, "caution", -26.9), 8: (4.9416, 13.1, 1.08, 2.25, "caution", -26.5),
       9: (4.9323, 13.1, 0.84, 3.19, "caution", -23.0), 10: (4.9123, 15.8, 0.78, 2.36, "caution", -34.3)}


@pytest.fixture(scope="module")
def scans():
    return ScanSet.from_folder(DATA).run()


def test_recovery(scans):
    for a in scans.analyses:
        assert a.recovery.gate_time == pytest.approx(4.0)
        assert a.recovery.integer_ok
        assert a.recovery.misfit.max() < 2e-4


@pytest.mark.parametrize("scan", range(1, 11))
def test_scan(scans, scan):
    a = scans.analyses[scan - 1]
    eth, se, chi, z, level, shift = REF[scan]
    assert a.result.estimate == pytest.approx(eth, abs=5e-5)
    assert a.result.estimate_se * 1e3 == pytest.approx(se, abs=0.06)
    assert a.result.chi2_red == pytest.approx(chi, abs=0.006)
    assert a.verdict_.z_low == pytest.approx(z, abs=0.006)
    assert a.verdict_.level == level
    if level == "caution":
        assert a.verdict_.cause == "background (falling)"
        assert a.verdict_.shift * 1e3 == pytest.approx(shift, abs=0.06)


def test_display_flat(scans):
    """Display yield below the onset scatters around zero (JSA 5.2)."""
    a = scans.analyses[0]
    chk = a.display_.flat_check(a.spectrum.x <= 4.60 + 1e-9)
    assert abs(chk["mean_z"]) < 1.5
    assert chk["negative"] < chk["points"]
