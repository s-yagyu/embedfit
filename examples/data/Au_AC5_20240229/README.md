# Au, Riken Keiki AC-5, 2024-02-29 (10 consecutive scans)

Rolled Au plate (99.99 %), the reference sample of Riken Keiki, measured with an AC-5
demonstration unit at NIMS on 2024-02-29. Photon energy 3.80–6.20 eV in 0.05 eV steps
(49 points), target light power 10 nW, anode voltage 2710 V, header dead time 0.00457 s,
header counting time 5 s (gate time 4 s). Scan starts are 7.4–7.5 min apart.

These are the data of the JSA paper and of the 1/n-Scan paper (also published in n-pys,
`sample_data/Au`, https://github.com/s-yagyu/n-pys). License: BSD 3-Clause, the same as n-pys.
Copyright (c) 2026, Shinjiro Yagyu (copyright notice of n-pys).

Columns of each `.dat`: photon energy (eV), count rate after the dark rate was subtracted
(cps), background-window flag, rise-window flag, light power. The light-correction file
named in the header (`10nW 240228144459.ldat`) is not included; its values are the light
power column.
