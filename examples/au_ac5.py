"""Example: the ten AC-5 Au scans of the JSA paper.

    python examples/au_ac5.py [output folder]

Writes the overview and display-yield figures of scan 1, the scan table and CSV files.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from embedfit import ScanSet  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
os.makedirs(out, exist_ok=True)

ss = ScanSet.from_folder(os.path.join(HERE, "data", "Au_AC5_20240229")).run()
print("scan  Eth (eV)  SE (meV)  chi2/dof  z_low  verdict")
for r in ss.table():
    extra = f" ({r['cause']}, {r['shift'] * 1e3:+.1f} meV)" if r["verdict"] == "caution" else ""
    print(f"{r['scan']:>4}  {r['estimate']:.4f}   {r['se'] * 1e3:5.1f}    {r['chi2_red']:.2f}   {r['z_low']:+.2f}  "
          f"{r['verdict']}{extra}")

an = ss.analyses[0]
print()
print(an.report_text())
an.plot("overview").savefig(os.path.join(out, "au_scan1_overview.png"), dpi=150)
an.plot("display").savefig(os.path.join(out, "au_scan1_display.png"), dpi=150)
an.plot("display", power=0.5).savefig(os.path.join(out, "au_scan1_display_sqrt.png"), dpi=150)
ss.plot().savefig(os.path.join(out, "au_scans.png"), dpi=150)
an.to_csv(out, "au_scan1")
ss.to_csv(os.path.join(out, "au_scans.csv"))
print(f"\nfigures and CSV written to {out}")
