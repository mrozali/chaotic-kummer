#!/usr/bin/env python3
"""Rerun the control of the unfolding (Supplemental Material, Sec. S4) and compare with the stored record.

The staircase of a sub-block with n levels is fitted by a Chebyshev polynomial of degree max(12, round(n/100)).  The
script runs code/kummer/r2g_unfolding_control.py in a sandbox.  That module unfolds the sub-blocks of the reference
direction at E = 6, 7, 8 and chiral GOE spectra of the same shapes, with fixed degrees 12, 20, 30, 50 (table A) and
with the rules degree 12 and max(12, n/K) for K = 200, 100, 60 (table B), and records the number variance and the
residual of the fitted staircase.

The fresh output is compared number by number with data/r2_out/Unfolding_control.json, with a tolerance of 0.01.  On
the machine that made the record the numbers are identical.  With another linear-algebra library the fits of the
highest degrees change the number variance in the third or fourth decimal, because the staircase fit of degree 61 on
the eigenspace of 6137 levels is close to the limit of double precision.  Then the statements of the Supplemental
Material are checked on the fresh output:
  * at degree 12 the fit of the larger sub-blocks leaves a slowly varying residual, which raises the number variance at
    large L,
  * chiral GOE spectra unfolded with the Letter's rule reproduce the GOE number variance for L <= 10 at E = 6, 7, 8,
  * at E = 6 the fit cannot be refined further without absorbing the fluctuations.

Run from the repository root:  .venv/bin/python code/figure-reproduction/check_unfolding_control.py    (about 2.5 minutes)
"""
import numpy as np

from checklib import J, Report, json_diff, run_module, sandbox

rep = Report("Control of the size-scaled unfolding")
out = sandbox("unfolding_control", ["Generic_firstorder_qgen_L7.npz", "Generic_probe_qgen_E8_mf_values.npz",
                                    "Hidden_firstorder_E8_values.npz"])
status, seconds, log = run_module("r2g_unfolding_control.py", ["--no-plot"], out)
rep.check("r2g_unfolding_control.py ran", status == 0, "%.0f s, log %s" % (seconds, log))
if status == 0:
    new, old = J("Unfolding_control.json", out), J("Unfolding_control.json")
    n, worst, other = json_diff(new, old)
    rep.check("equal to the stored record within 0.01", worst < 1e-2 and not other,
              "%d numbers, largest difference %.1e%s" % (n, worst, (", " + "; ".join(other[:3])) if other else ""))
    goe = np.array(new["GOE"][:4])                     # GOE number variance at L = 1, 2, 5, 10
    letter, fixed = new["B"]["degree max(12, n/100)"], new["B"]["degree 12"]
    for E in ("7", "8"):
        rep.check("E = %s: degree 12 leaves a larger residual and a larger number variance at L = 10" % E,
                  min(fixed[E]["residuals"]) > 2 * max(letter[E]["residuals"][:4]) and fixed[E]["data"][3] > letter[E]["data"][3] + 0.05,
                  "residuals %s against %s levels, Sigma^2(10) %.3f against %.3f"
                  % (np.round(fixed[E]["residuals"], 1), np.round(letter[E]["residuals"], 1), fixed[E]["data"][3], letter[E]["data"][3]))
    for E in ("6", "7", "8"):
        ctrl = np.array(letter[E]["goe"][:4])
        rep.check("E = %s: chiral GOE control reproduces the GOE number variance for L <= 10" % E, np.max(np.abs(ctrl - goe)) < 0.02,
                  "control %s, GOE %s, degrees %s" % (np.round(ctrl, 3), np.round(goe, 3), letter[E]["degrees"]))
    a451, a899 = new["A"]["E=6, eigenspace 451"], new["A"]["E=6, eigenspace 899"]
    rep.check("E = 6: higher degrees absorb fluctuations (chiral GOE control falls below the GOE at L = 10)",
              a451["30"]["goe"][3] < goe[3] - 0.08 and a451["50"]["goe"][3] < a451["30"]["goe"][3] and a899["50"]["goe"][3] < goe[3] - 0.05,
              "451 levels: %.3f at degree 30, %.3f at degree 50.  899 levels: %.3f at degree 50.  GOE %.3f"
              % (a451["30"]["goe"][3], a451["50"]["goe"][3], a899["50"]["goe"][3], goe[3]))
rep.finish()
