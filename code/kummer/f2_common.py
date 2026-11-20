"""Shared RMT utilities for F2 (random-matrix null model).

GOE reference formulas, unfolding via an ensemble-averaged smoothed density,
gap-ratio statistic, number variance, and the (unfolded / windowed) connected
spectral form factor.  Used by f2a_run.py and f2b_run.py
(not included in this repository).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------
# Reference curves
# ----------------------------------------------------------------------

def goe_b(tau):
    """GOE connected SFF shape b(tau), unfolded units (mean spacing 1),
    tau = t/(2 pi).  b(tau) -> plateau value 1 as tau -> infinity."""
    tau = np.asarray(tau, dtype=float)
    out = np.empty_like(tau)
    small = tau <= 1.0
    t = tau[small]
    out[small] = 2 * t - t * np.log(1 + 2 * t)
    t = tau[~small]
    out[~small] = 2 - t * np.log((2 * t + 1) / (2 * t - 1))
    return out


def poisson_b(tau):
    """Poisson connected SFF shape: flat plateau at 1 for all tau > 0
    (no ramp; included for reference on SFF panels)."""
    return np.ones_like(np.asarray(tau, dtype=float))


def wigner_surmise_goe(s):
    s = np.asarray(s, dtype=float)
    return (np.pi / 2.0) * s * np.exp(-np.pi * s ** 2 / 4.0)


def poisson_ps(s):
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


def goe_number_variance(L):
    """Sigma^2(L) ~ (2/pi^2)(ln(2 pi L) + gamma + 1 - pi^2/8), GOE, unfolded units."""
    L = np.asarray(L, dtype=float)
    gamma = 0.5772156649015329
    return (2.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + gamma + 1 - np.pi ** 2 / 8.0)


def poisson_number_variance(L):
    return np.asarray(L, dtype=float)


MEAN_R_GOE = 0.5307
MEAN_R_POISSON = 0.3863

# ----------------------------------------------------------------------
# Gap-ratio statistic (no unfolding needed)
# ----------------------------------------------------------------------

def gap_ratios(levels_sorted):
    """r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) for consecutive spacings
    of a sorted 1d array of levels.  Returns array of length len(levels)-2."""
    levels_sorted = np.asarray(levels_sorted, dtype=float)
    d = np.diff(levels_sorted)
    d = d[d > 0]  # guard against accidental exact ties
    if len(d) < 2:
        return np.array([])
    r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
    return r


def bulk_slice(n, frac=0.1):
    """Index slice dropping frac at each edge of a length-n sorted array."""
    lo = int(np.floor(frac * n))
    hi = n - lo
    return slice(lo, hi)

# ----------------------------------------------------------------------
# Unfolding: ensemble-averaged smoothed density via a Gaussian-filtered
# pooled histogram of the (positive) spectrum, giving a smooth estimate of
# the ensemble-averaged cumulative counting function N(E); unfold a given
# sample's levels E_i via xi_i = Nhat(E_i).
# ----------------------------------------------------------------------

def fit_avg_counting_function(pooled_values, n_samples, nbins=400, smooth_bins=3.0):
    """Fit Nhat(E) = ensemble-averaged number of levels <= E, from pooled
    (across-sample) values.  Returns a callable Nhat(E) (monotonic
    piecewise-linear interpolant) built from a Gaussian-smoothed histogram.
    """
    pooled_values = np.asarray(pooled_values, dtype=float)
    lo, hi = pooled_values.min(), pooled_values.max()
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    edges = np.linspace(lo - pad, hi + pad, nbins + 1)
    counts, _ = np.histogram(pooled_values, bins=edges)
    counts = counts.astype(float) / n_samples
    counts_smooth = gaussian_filter1d(counts, sigma=smooth_bins, mode="nearest")
    # renormalize so total count is preserved after smoothing
    if counts_smooth.sum() > 0:
        counts_smooth *= counts.sum() / counts_smooth.sum()
    cum = np.concatenate([[0.0], np.cumsum(counts_smooth)])

    def Nhat(E):
        return np.interp(E, edges, cum)

    return Nhat


def fit_avg_density(pooled_values, n_samples, nbins=400, smooth_bins=3.0):
    """Companion to fit_avg_counting_function: returns a callable rho(E) =
    ensemble-averaged smoothed level density (levels per unit E per sample),
    built from the same Gaussian-filtered pooled histogram."""
    pooled_values = np.asarray(pooled_values, dtype=float)
    lo, hi = pooled_values.min(), pooled_values.max()
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    edges = np.linspace(lo - pad, hi + pad, nbins + 1)
    counts, _ = np.histogram(pooled_values, bins=edges)
    counts = counts.astype(float) / n_samples
    counts_smooth = gaussian_filter1d(counts, sigma=smooth_bins, mode="nearest")
    if counts_smooth.sum() > 0:
        counts_smooth *= counts.sum() / counts_smooth.sum()
    binw = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = counts_smooth / binw

    def rho(E):
        return np.interp(E, centers, dens, left=dens[0], right=dens[-1])

    return rho


def unfold_samples(samples, Nhat):
    """Apply xi_i = Nhat(E_i) to each sample (list of 1d arrays); returns a
    list of unfolded arrays (mean spacing ~1 in the bulk)."""
    return [np.sort(Nhat(np.asarray(s, dtype=float))) for s in samples]

# ----------------------------------------------------------------------
# Number variance (unfolded, bulk, ensemble + sliding-window average)
# ----------------------------------------------------------------------

def number_variance(unfolded_samples, Ls, n_windows=40, margin=None):
    """Sigma^2(L) for each L in Ls, averaged over sliding windows within the
    bulk range of each sample and over samples.  unfolded_samples: list of
    1d sorted arrays (already restricted to the bulk if desired).  margin:
    extra guard (in unfolded units) kept clear at each edge when placing
    window centers (defaults to max(Ls))."""
    Ls = np.asarray(Ls, dtype=float)
    if margin is None:
        margin = Ls.max()
    out = np.zeros(len(Ls))
    out2 = np.zeros(len(Ls))
    ntot = 0
    for xi in unfolded_samples:
        if len(xi) < 5:
            continue
        lo, hi = xi.min(), xi.max()
        centers = np.linspace(lo + margin, hi - margin, n_windows)
        if hi - margin <= lo + margin:
            continue
        for j, L in enumerate(Ls):
            left = centers - L / 2.0
            right = centers + L / 2.0
            # counts via searchsorted (vectorized over window centers)
            n_left = np.searchsorted(xi, left, side="left")
            n_right = np.searchsorted(xi, right, side="left")
            n_in = (n_right - n_left).astype(float)
            out[j] += n_in.sum()
            out2[j] += (n_in ** 2).sum()
        ntot += len(centers)
    if ntot == 0:
        return np.full(len(Ls), np.nan)
    mean = out / ntot
    mean2 = out2 / ntot
    return mean2 - mean ** 2

# ----------------------------------------------------------------------
# Connected spectral form factor (unfolded units), Gaussian window in the
# bulk, ensemble-averaged, disconnected part subtracted.
# ----------------------------------------------------------------------

def sff_unfolded(unfolded_samples, taus, window_sigma_frac=0.35):
    """Connected SFF K(tau) on unfolded spectra, tau = t/(2 pi) so that
    K(tau) -> plateau = sum_i w(xi_i)^2 as tau -> infinity for GOE, matching
    goe_b(tau) after dividing by that plateau.  window_sigma_frac sets the
    Gaussian window width as a fraction of the half-range of each sample's
    bulk span.  Returns (K(tau) array, plateau estimate, ts array actually
    used = 2 pi tau)."""
    taus = np.asarray(taus, dtype=float)
    ts = 2 * np.pi * taus
    amp_sum = np.zeros(len(ts), dtype=complex)
    amp2_sum = np.zeros(len(ts), dtype=float)
    plateau_sum = 0.0
    n = 0
    for xi in unfolded_samples:
        if len(xi) < 5:
            continue
        c = 0.5 * (xi.min() + xi.max())
        half = 0.5 * (xi.max() - xi.min())
        sig = window_sigma_frac * half
        w = np.exp(-0.5 * ((xi - c) / sig) ** 2)
        phase = np.exp(-1j * np.outer(ts, xi))  # (len(ts), len(xi))
        amp = phase @ w  # complex, shape (len(ts),)
        amp_sum += amp
        amp2_sum += np.abs(amp) ** 2
        plateau_sum += np.sum(w ** 2)
        n += 1
    if n == 0:
        return np.full(len(ts), np.nan), np.nan
    disc = np.abs(amp_sum / n) ** 2
    conn = amp2_sum / n - disc
    plateau = plateau_sum / n
    return conn, plateau


def sff_energy_domain(samples_energies, ts, E_center, window_sigma):
    """Connected SFF directly in physical energy units (no unfolding), with
    a Gaussian window w(E) = exp(-(E-E_center)^2 / (2 window_sigma^2))
    centered at E_center.  Used for Experiment B.  Returns (K_c(t), plateau
    = <sum w(E_i)^2>)."""
    ts = np.asarray(ts, dtype=float)
    amp_sum = np.zeros(len(ts), dtype=complex)
    amp2_sum = np.zeros(len(ts), dtype=float)
    plateau_sum = 0.0
    n = 0
    for E in samples_energies:
        E = np.asarray(E, dtype=float)
        w = np.exp(-0.5 * ((E - E_center) / window_sigma) ** 2)
        phase = np.exp(-1j * np.outer(ts, E))
        amp = phase @ w
        amp_sum += amp
        amp2_sum += np.abs(amp) ** 2
        plateau_sum += np.sum(w ** 2)
        n += 1
    if n == 0:
        return np.full(len(ts), np.nan), np.nan
    disc = np.abs(amp_sum / n) ** 2
    conn = amp2_sum / n - disc
    plateau = plateau_sum / n
    return conn, plateau


def goe_sff_energy_domain(ts, E_center, window_sigma, rho_func, E_lo, E_hi, n_grid=4001):
    """GOE prediction K_GOE(t) = int dE w(E)^2 rho(E) b(t / (2 pi rho(E))),
    no fit parameters, rho_func(E) the smoothed level density (levels per
    unit energy)."""
    ts = np.asarray(ts, dtype=float)
    Egrid = np.linspace(E_lo, E_hi, n_grid)
    w2 = np.exp(-((Egrid - E_center) ** 2) / (window_sigma ** 2))
    rho = np.clip(rho_func(Egrid), 1e-12, None)
    integrand_base = w2 * rho  # function of E only
    out = np.empty(len(ts))
    dE = Egrid[1] - Egrid[0]
    for i, t in enumerate(ts):
        tau = t / (2 * np.pi * rho)
        b = goe_b(tau)
        out[i] = np.sum(integrand_base * b) * dE
    return out, Egrid, rho
