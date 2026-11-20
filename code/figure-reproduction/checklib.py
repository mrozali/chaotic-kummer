"""Helpers shared by the check scripts of this folder.

Records are read from data/r2_out at the repository root, or from the folder named by the environment variable
KUMMER_OUT.  Check scripts that run a pipeline module do so in a sandbox under code/figure-reproduction/generated/,
so that the stored records are never overwritten.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as cheb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # the repository root
R2 = os.environ.get("KUMMER_OUT") or os.path.join(ROOT, "data", "r2_out")
KUMMER = os.path.join(ROOT, "code", "kummer")
GEN = os.path.join(HERE, "generated")
R_GOE = 0.5307
RANDOM20 = ["qrand%d" % k for k in range(101, 121)]              # the 20 random directions of Fig. 2


def J(name, folder=None):
    with open(os.path.join(folder or R2, name)) as fh:
        return json.load(fh)


def Z(name, folder=None):
    return np.load(os.path.join(folder or R2, name), allow_pickle=False)


def distinct(v):
    """Non-zero values, distinct to 1e-9, ascending (the convention of every record)."""
    v = np.asarray(v, dtype=float)
    v = v[v > 1e-9 * max(v.max(initial=0.0), 1e-30)]
    return np.unique(np.round(v, 9))


def gap_ratios(v):
    """r_i = min(d_i, d_i+1) / max(d_i, d_i+1) of the sorted values, eq. (r) of the Letter."""
    d = np.diff(np.sort(np.asarray(v, dtype=float)))
    return np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])


def pooled(blocks, nmin=3):
    """Mean gap ratio over the ratios of all sub-blocks with at least nmin distinct values: (mean, sem, number)."""
    g = np.concatenate([gap_ratios(distinct(b)) for b in blocks if distinct(b).size >= nmin])
    return float(g.mean()), float(g.std() / np.sqrt(g.size)), int(g.size)


def unfold(x, deg=None, trim=0.05):
    """Unfolded levels of one sub-block as in the Supplemental Material: Chebyshev fit of the staircase of degree
    max(12, round(n/100)), 5% dropped at each end.  Returns (levels, degree, monotone on the retained range)."""
    x = np.unique(np.round(np.asarray(x, dtype=float), 9))
    n = x.size
    if deg is None:
        deg = max(12, int(round(n / 100.0)))
    t = (2.0 * x - x[0] - x[-1]) / (x[-1] - x[0])
    c, (_, rank, _, _) = cheb.chebfit(t, np.arange(1, n + 1), deg, rcond=1e-15, full=True)
    assert rank == deg + 1, (n, deg, rank)
    xi = cheb.chebval(t, c)
    k = int(trim * n)
    tt = np.linspace(t[k], t[n - k - 1], 40000)
    return xi[k:n - k], deg, bool(np.all(cheb.chebval(tt, cheb.chebder(c)) > 0))


def number_variance(xi, Ls, npos=4000):
    out = []
    for L in Ls:
        a = np.linspace(xi[0], xi[-1] - L, npos)
        out.append(np.var(np.searchsorted(xi, a + L) - np.searchsorted(xi, a)))
    return np.array(out)


def sigma2_goe(L):
    """Number variance of the GOE, eq. (sigma2goe) of the Supplemental Material."""
    from scipy.special import sici
    L = np.asarray(L, dtype=float)
    si2, ci2 = sici(2 * np.pi * L)
    si1, _ = sici(np.pi * L)
    gue = (np.log(2 * np.pi * L) + np.euler_gamma + 1 - np.cos(2 * np.pi * L) - ci2) / np.pi ** 2 + L * (1 - 2 / np.pi * si2)
    return 2 * gue + si1 ** 2 / np.pi ** 2 - si1 / np.pi


def mirror_angle(q):
    """Angle in degrees from the unit quaternion q to the nearest of the 16 mirrors q_mu = 0, q_mu = +-q_nu."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    s = [abs(q[m]) for m in range(4)]
    s += [abs(q[m] - sg * q[n]) / np.sqrt(2) for m in range(4) for n in range(m + 1, 4) for sg in (1, -1)]
    return float(np.degrees(np.arcsin(min(1.0, min(s)))))


class Report:
    """Collects checks.  Prints one line per check and at the end 'RESULT: PASS' or 'RESULT: FAIL'."""

    def __init__(self, title):
        self.n = self.bad = 0
        self.t0 = time.time()
        print(title)

    def section(self, text):
        print("\n" + text)

    def check(self, label, ok, detail=""):
        self.n += 1
        self.bad += 0 if ok else 1
        print("  %s  %s%s" % ("ok  " if ok else "FAIL", label, (": " + detail) if detail else ""))
        return bool(ok)

    def value(self, label, value, paper):
        """The value rounded to the digits printed in the paper must equal the paper's string."""
        digits = len(paper.split(".")[1]) if "." in paper else 0
        got = "%.*f" % (digits, value)
        return self.check(label, got == paper, "%.6g, paper %s" % (value, paper))

    def integer(self, label, value, paper):
        return self.check(label, int(value) == int(paper), "%d, paper %d" % (value, paper))

    def info(self, text):
        print("        " + text)

    def finish(self):
        print("\n%d checks, %d failed  [%.0f s]" % (self.n, self.bad, time.time() - self.t0))
        print("RESULT: PASS" if self.bad == 0 else "RESULT: FAIL")
        sys.exit(1 if self.bad else 0)


def sandbox(name, inputs=()):
    """A fresh record folder generated/<name>/r2_out holding copies of the named stored records."""
    d = os.path.join(GEN, name, "r2_out")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in inputs:
        os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
        shutil.copy(os.path.join(R2, f), os.path.join(d, f))
    return d


def run_module(script, args, out_dir, threads=None):
    """Run code/kummer/<script> with the given arguments, its record folder redirected to out_dir.
    The output goes to <out_dir>/../<script>.log.  Returns (exit status, seconds, path of the log from the root)."""
    env = dict(os.environ, KUMMER_OUT=out_dir, PYTHONDONTWRITEBYTECODE="1")
    if threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[v] = str(threads)
    log = os.path.join(os.path.dirname(out_dir), os.path.splitext(script)[0] + ".log")
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, "-B", os.path.join(KUMMER, script)] + [str(a) for a in args],
                           cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0, os.path.relpath(log, ROOT)


def match_blocks(new, old, tol=2e-9):
    """Compare two sets of spectra (dicts key -> distinct values) as sets: every new array must equal one old array.
    Returns (largest deviation of the best matching, whether the keys also agree).  The matching by content makes
    the comparison independent of the order in which a linear-algebra library returns the characters."""
    if len(new) != len(old):
        return float("inf"), False
    left = {k: np.sort(np.asarray(w, dtype=float)) for k, w in old.items()}
    worst, same_keys = 0.0, True
    for k, v in new.items():
        v = np.sort(np.asarray(v, dtype=float))
        dev = {k2: (float(np.max(np.abs(w - v))) if v.size else 0.0) for k2, w in left.items() if w.size == v.size}
        if not dev:
            return float("inf"), False
        pick = k if dev.get(k, float("inf")) <= tol else min(dev, key=dev.get)      # the same key when it matches
        same_keys = same_keys and pick == k
        worst = max(worst, dev[pick])
        del left[pick]
    return worst, same_keys


def json_diff(a, b, path=""):
    """Compare two nested JSON objects.  Returns (number of numbers compared, largest absolute difference,
    list of structural differences)."""
    n, worst, other = 0, 0.0, []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            other.append(path + ": keys differ")
        for k in a:
            if k in b:
                m, w, o = json_diff(a[k], b[k], path + "/" + str(k))
                n, worst, other = n + m, max(worst, w), other + o
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            other.append(path + ": lengths differ")
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                m, w, o = json_diff(u, v, path + "[%d]" % i)
                n, worst, other = n + m, max(worst, w), other + o
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        n, worst = 1, abs(a - b)
    elif a != b:
        other.append(path + ": %r against %r" % (a, b))
    return n, worst, other
