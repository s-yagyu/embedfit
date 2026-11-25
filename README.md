# embedfit

日本語版: [README_jpn.md](README_jpn.md)

Analysis of spectra normalized by a reference quantity, with the instrument's
corrections embedded in the model.

Many spectrometers output a corrected value: the recorded counts minus a dark
level, passed through a detector correction G (dead time, efficiency) and
divided by a reference quantity Phi (photon flux, I0). Fitting a physical model
to that corrected value bends the background and the noise. `embedfit` turns
the procedure around and puts the corrections into the model, then compares the
model with the recorded counts by the Poisson likelihood:

```
lambda = d + c Phi + k Phi f(x; theta)        true rate
mu_i   = T_i G^{-1}(lambda_i)                 expected recorded counts
N_i    ~ Poisson(mu_i)
```

From one spectrum this gives the estimate (e.g. the photoemission threshold),
its standard error, and a verdict on whether the model describes the data
(normal / caution / abnormal). The method needs the recorded counts, or an
output from which they can be recovered, and the instrument's correction
formula with its constants.

Reference: S. Yagyu, T. Nagata, and Y. Nakajima, "Analysis of counting data
normalized by a reference quantity: embedding the instrument corrections in the
model, with application to photoelectron yield spectroscopy" (submitted to the
Journal of Surface Analysis; preprint on Jxiv). Please cite this paper when you
use the package.

## Install

```
pip install .            # numpy, scipy
pip install ".[plot]"      # + matplotlib for the figures
```

## Quick start (Riken Keiki AC series)

```python
from embedfit import Analysis, ScanSet

an = Analysis.from_file("examples/data/Au_AC5_20240229/Au 240229034207_1.dat").run()
print(an.report_text())         # recovery of counts, fit, verdict, display yield
an.plot("overview")             # 2 x 3 overview figure
an.plot("display", power=0.5)   # display yield; Y^(1/2) applied to the model curve only
an.to_csv("out")                # the values behind the figures, and a JSON report

scans = ScanSet.from_folder("examples/data/Au_AC5_20240229").run()
for row in scans.table():
    print(row["scan"], row["estimate"], row["se"], row["verdict"])
```

`examples/au_ac5.py` reproduces the ten Au scans of the paper.

## What `run()` does

| Step | Function | Paper |
|---|---|---|
| Read the file and record what the instrument did (G, Phi, dark subtraction) | `readers.read_ac` | Sec. 4.2 |
| Recover integer counts from the dark-subtracted rate (path B) | `recover_counts` | Sec. 3.4.1, App. B.1 |
| Fit the forward model by the Poisson likelihood; SE from the Fisher information | `fit` | Sec. 3.3, App. B.3 |
| chi2/dof and z_low; control models (c free, n free); verdict | `verdict` | Sec. 3.5 |
| Display yield (G(N/T) - d_hat)/Phi, with sigma_y | `display_yield` | Sec. 4.3, App. B.4 |

## Design

The model is assembled from parts that vary independently between techniques,
instead of one subclass per technique:

| Part | Module | Provided |
|---|---|---|
| Data and what the instrument did | `data.Spectrum` | signal kinds `counts`, `rate`, `rate_dark_subtracted`, `current` |
| Reader (one per file format) | `readers` | Riken Keiki AC series, generic table/arrays |
| Detector correction G | `detector` | `RikenAC`, `NonParalyzable`, `Identity` |
| Signal shape f | `model` | `PowerThreshold` |
| Background terms | `model` | `Constant` (d), `FluxProportional` (c Phi), `ExpDecay`, `PowerDecay`, `LinearDrift` |
| Noise model | `noise` | `Poisson`, `Gaussian` |
| Presets (a technique = a choice of parts) | `presets` | `pys_counting`, `pys_current` (experimental) |

A new technique is a new preset; a new instrument is a new reader (register it
with `readers.register`). For example, inverse photoemission shares the
counting noise and G with counting PYS and needs only another signal shape;
current-method PYS shares the signal shape and needs the Gaussian noise model.

## Status

* **Counting PYS (AC series)**: reproduces the JSA paper (tests in `tests/test_au.py`).
* **Current-method PYS**: *experimental*. Checked on synthetic data only; the
  electrometer filter is not modelled; the noise parameters sigma0 and rel must
  be measured separately.
* Other techniques (transmission, inverse photoemission, XPS/UPS, NEXAFS) are
  not implemented. The parts above are meant to cover them.
* The package does not include the automated conventional method (1/n-Scan,
  published in n-pys).

## Tests

```
pip install ".[test]"
pytest
```

## License

BSD 3-Clause (see `LICENSE`). The sample data in `examples/data` are also under the BSD 3-Clause License (the same as n-pys, where they are published; the original copyright notice is in `examples/data/Au_AC5_20240229/README.md`).
