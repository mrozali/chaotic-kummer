#!/usr/bin/env python3
"""Rerun the analysis behind the E = 4 sentences of the End Matter and compare with the stored record.

The Letter states that the low gap ratio at E = 4 is a property of the level and not of the reference direction.  The
script runs code/kummer/r2g_E4_deficit.py in a sandbox.  That module builds the first-order matrices of E = 4 once,
evaluates the sub-block spectra for 3000 random directions more than 4 degrees from every mirror (three seeds, 1000
directions each), finds the hidden Z_2 of the trivial character from two directions, follows the top of the spectrum
from the singlet direction, and repeats the top-of-spectrum statistic at E = 3 and 5.

The fresh output is compared number by number with data/r2_out/E4_deficit.json.  The paper values themselves are
checked against that record by check_paper_numbers.py, section (F).  The stored record also holds E = 6, from a longer
run with the options --levels=3,5,6 --nother=150 (about 11 minutes), which this check does not repeat.

Run from the repository root:  .venv/bin/python code/figure-reproduction/check_E4_deficit.py      (about 70 s, 1.1 GB)
"""
from checklib import J, Report, json_diff, run_module, sandbox

rep = Report("E = 4: the gap-ratio deficit is a property of the level")
out = sandbox("E4_deficit")
status, seconds, log = run_module("r2g_E4_deficit.py", [], out)
rep.check("r2g_E4_deficit.py ran", status == 0, "%.0f s, log %s" % (seconds, log))
if status == 0:
    new, old = J("E4_deficit.json", out), J("E4_deficit.json")
    for key in ("reference", "samples", "all_samples", "joint_invariant_subspaces", "singlet_top", "top_positions"):
        n, worst, other = json_diff(new[key], old[key])
        rep.check("'%s' equal to the stored record" % key, worst < 1e-9 and not other,
                  "%d numbers, largest difference %.1e%s" % (n, worst, (", " + "; ".join(other[:3])) if other else ""))
    al = new["all_samples"]
    rep.info("3000 directions: pooled <r> %.4f, spread %.4f, top ten ratios %.3f, remaining %.3f, eigenspaces %.3f and %.3f"
             % (al["pooled_mean"], al["pooled_sd_between_directions"], al["clean_top10"], al["clean_rest"], al["leaf50"], al["leaf26"]))
    for E in ("3", "5"):
        a, b = new["other_levels"][E], old["other_levels"][E]
        rep.check("E = %s: no deficit at the top of the spectrum" % E, a["top10"] > 0.48 and abs(a["top10"] - b["top10"]) < 0.02,
                  "top ten %.3f, rest %.3f over %d directions (record: %.3f, %.3f over %d)"
                  % (a["top10"], a["rest"], a["n_directions"], b["top10"], b["rest"], b["n_directions"]))
rep.finish()
