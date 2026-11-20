"""Shared style, paths and random-matrix helpers of the figure scripts in this folder.

Every figure script imports this module.  The records are read from data/r2_out at the repository root, or from the
folder named by the environment variable KUMMER_OUT.  Figures are written to code/figure-reproduction/generated/, which
is not committed.  save() then compares the new PDF with the Letter's figure in paper/ and prints RESULT: PASS
or RESULT: FAIL.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # the repository root
R2 = os.environ.get("KUMMER_OUT") or os.path.join(ROOT, "data", "r2_out")
GEN = os.path.join(HERE, "generated")

# ---------------------------------------------------------------- style 
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
BLUE = "#2a78d6"      # first-order (split) levels, the reference direction
ORANGE = "#eb6834"    # first-order kernel
INK = "#0b0b0b"       # primary ink, reference curves
INK2 = "#52514e"      # secondary text
GRAY = "#898781"      # muted marks (unresolved, random directions)
LIGHT = "#c3c2b7"     # hairlines
SHADE = "#f0efec"     # shaded ranges
GREEN = "#1f9e6e"     # the singlet direction
GREEN_LIGHT = "#76c7a3"   # lighter step of GREEN, for a lower level of the singlet
EN = "{"              # the en dash: matplotlib's cmr10 is in TeX text encoding, where "{" is the en dash

# ---------------------------------------------------------------- random-matrix references
R_GOE, R_GUE, R_POI, R_TWO = 0.5307, 0.5996, 0.3863, 0.423     # values quoted in the paper


def gap_ratios(values):
    """r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}) of the sorted values (the paper's eq:r)."""
    s = np.sort(np.asarray(values, dtype=float))
    d = np.diff(s)
    return np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])


_ZB = {1: 8.0 / 27.0, 2: 4.0 * np.pi / (81.0 * np.sqrt(3.0))}


def surmise(r, beta):
    """Wigner-like surmise of Atas, Bogomolny, Giraud and Roux for r in [0, 1]."""
    r = np.asarray(r, dtype=float)
    return 2.0 / _ZB[beta] * (r + r * r) ** beta / (1.0 + r + r * r) ** (1.0 + 1.5 * beta)


def poisson(r):
    return 2.0 / (1.0 + np.asarray(r, dtype=float)) ** 2


def ks_distance(sample, pdf):
    """Kolmogorov-Smirnov distance of a sample on [0, 1] from a density given as a function."""
    xs = np.linspace(0.0, 1.0, 20001)
    ps = pdf(xs)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (ps[1:] + ps[:-1]) * np.diff(xs))])
    cdf /= cdf[-1]
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    F = np.interp(x, xs, cdf)
    return max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))


def compare_with_paper(name, dpi=100, tol=16, max_fraction=1e-3):
    """Render generated/<name>.pdf and paper/<name>.pdf with pdftoppm and compare them pixel by pixel.

    Returns True if the images have the same size and at most max_fraction of the pixels differ by more than tol/255
    in any channel, False if not, and None if pdftoppm (Poppler) is not installed.
    """
    import matplotlib.image as mpimg
    ref = os.path.join(ROOT, "paper", name + ".pdf")
    if shutil.which("pdftoppm") is None:
        print("pixel comparison with the Letter's figure skipped: pdftoppm (Poppler) not found, the figure is not checked")
        return None
    img = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, pdf in (("ref", ref), ("new", os.path.join(GEN, name + ".pdf"))):
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, os.path.join(tmp, tag)], check=True)
            img.append(np.asarray(mpimg.imread(os.path.join(tmp, tag + ".png")), dtype=float)[..., :3])
    a, b = img
    if a.shape != b.shape:
        print(f"pixel comparison: image sizes differ, {b.shape[1]} x {b.shape[0]} against {a.shape[1]} x {a.shape[0]} in the Letter")
        return False
    ndiff = int(np.sum(np.abs(a - b).max(axis=-1) > tol / 255.0))
    npix = a.shape[0] * a.shape[1]
    print(f"pixel comparison with paper/{name}.pdf at {dpi} dpi: {ndiff} of {npix} pixels differ")
    return ndiff <= max_fraction * npix


def save(fig, script_file, name):
    """Write generated/<name>.pdf and .png, compare with the Letter's figure of the same name, print the verdict."""
    os.makedirs(GEN, exist_ok=True)
    out = os.path.join(GEN, name)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print("wrote", os.path.relpath(out + ".pdf", ROOT), "and .png")
    ok = compare_with_paper(name)
    print("RESULT: FAIL" if ok is False else "RESULT: PASS")
    if ok is False:
        sys.exit(1)
