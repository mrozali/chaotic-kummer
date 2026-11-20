"""Route 2 Phase C: level statistics of the truncated Hamiltonian.

Conventions follow `f2_common.py` (the F2 null model) so that the numbers are directly
comparable: energy-domain connected SFF with a Gaussian window w(E) = exp(-(E-E0)^2/2sig^2),
GOE prediction K_GOE(t) = int dE w^2 rho b(t/(2 pi rho)) with NO fit parameters, and the ramp
coefficient f = (through-origin slope of K_c over t <= 0.2 t_H) / (sig/sqrt(pi)).
The same estimator applied to the exact GOE curve returns f_ref ~ 0.85 (the well-known
downward bias of a through-origin fit over a finite early-time range); `f_cal = f / f_ref`
is quoted alongside.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import f2_common as f2


def smooth_density(E, nbins=60, smooth=2.0):
    """Gaussian-smoothed density (levels per unit E) of ONE spectrum, as a callable."""
    E = np.asarray(E, float)
    lo, hi = E.min(), E.max()
    pad = 0.02 * (hi - lo) + 1e-9
    edges = np.linspace(lo - pad, hi + pad, nbins + 1)
    c, _ = np.histogram(E, bins=edges)
    c = gaussian_filter1d(c.astype(float), smooth, mode='nearest')
    s = c.sum()
    if s > 0:
        c *= len(E) / s
    bw = edges[1] - edges[0]
    cen = 0.5 * (edges[:-1] + edges[1:])
    dens = c / bw
    cum = np.concatenate([[0.0], np.cumsum(c)])

    def rho(x):
        return np.interp(x, cen, dens, left=dens[0], right=dens[-1])

    def Nhat(x):
        return np.interp(x, edges, cum)

    return rho, Nhat


def sff_one(E, ts, E0, sig, rho):
    """Connected SFF of ONE spectrum with the smooth part subtracted analytically."""
    E = np.asarray(E, float)
    w = np.exp(-0.5 * ((E - E0) / sig) ** 2)
    amp = np.exp(-1j * np.outer(ts, E)) @ w
    # smooth (disconnected) part from the fitted density.  The grid must resolve the phase
    # exp(-i t g) at the largest time requested: with a fixed 2001-point grid the Riemann sum
    # aliases at t* = 2 pi / dg, where it returns the full window weight and Kc drops by -W^2
    # (seen in the 11k-state blocks of the weight-one sector at Lambda = 4, where
    # 0.2 t_H exceeds t*).  We keep t* >= 4 t_max, which leaves every earlier result unchanged
    # (the disconnected part is negligible for t sig >> 1 apart from the alias).
    ngrid = max(2001, int(4.0 * np.max(np.abs(ts)) * (E.max() - E.min()) / (2 * np.pi)) + 1)
    g = np.linspace(E.min(), E.max(), ngrid)
    wg = np.exp(-0.5 * ((g - E0) / sig) ** 2) * rho(g)
    dsc = (np.exp(-1j * np.outer(ts, g)) @ wg) * (g[1] - g[0])
    return np.abs(amp) ** 2 - np.abs(dsc) ** 2, float(np.sum(w ** 2))


def sff_goe(ts, E0, sig, rho, Elo, Ehi, n=2001):
    g = np.linspace(Elo, Ehi, n)
    w2 = np.exp(-((g - E0) ** 2) / sig ** 2)
    r = np.clip(rho(g), 1e-12, None)
    base = w2 * r
    dg = g[1] - g[0]
    out = np.empty(len(ts))
    for i, t in enumerate(ts):
        out[i] = np.sum(base * f2.goe_b(t / (2 * np.pi * r))) * dg
    return out


def slope_through_origin(ts, K):
    ok = np.isfinite(K)
    return float(np.sum(ts[ok] * K[ok]) / np.sum(ts[ok] ** 2))


def analyse(spectra, E0, half, tau_fit=0.2, nfit=15, shifts=(-0.15, 0.0, 0.15),
            Ls=(1, 2, 3, 5, 8, 12), ts_plot=None, bare_levels=None, poisson=False,
            seed=0):
    """`spectra` = list of 1d arrays (one per fully resolved sub-sector).
    Returns a dict of statistics, ensemble averaged over the sub-sectors and over the
    small window shifts."""
    sig = half / 2.0
    rng = np.random.default_rng(seed)
    rhos, Nhats, wins, used = [], [], [], []
    for E in spectra:
        if len(E) < 20:
            rhos.append(None)
            Nhats.append(None)
            wins.append(np.zeros(0))
            used.append(np.asarray(E, float))
            continue
        rho, Nh = smooth_density(E)
        if poisson:
            # i.i.d. resampling with the SAME smoothed density
            # continuous inverse-CDF sampling (a discrete grid would create exact ties
            # and bias <r> upward, since zero spacings are dropped)
            g = np.linspace(E.min(), E.max(), 8001)
            c = np.concatenate([[0.0], np.cumsum(np.clip(rho(g[:-1]), 0, None)
                                                 * np.diff(g))])
            c /= c[-1]
            E = np.sort(np.interp(rng.random(len(E)), c, g))
            rho, Nh = smooth_density(E)
        rhos.append(rho)
        Nhats.append(Nh)
        used.append(E)
        w = np.sort(E[(E >= E0 - half) & (E <= E0 + half)])
        wins.append(w)
    # ---- gap ratio (pooled), raw and after unfolding -----------------------
    rr = [f2.gap_ratios(w) for w in wins if len(w) > 5]
    rr = np.concatenate(rr) if rr else np.array([np.nan])
    ru = [f2.gap_ratios(np.sort(Nhats[i](wins[i]))) for i in range(len(wins))
          if Nhats[i] is not None and len(wins[i]) > 5]
    ru = np.concatenate(ru) if ru else np.array([np.nan])
    # ---- number variance ----
    unf = [np.sort(Nhats[i](wins[i])) for i in range(len(wins))
           if Nhats[i] is not None and len(wins[i]) > 8]
    Ls = np.asarray(Ls, float)
    sig2 = f2.number_variance(unf, Ls, margin=max(1.0, Ls.max() / 2)) if unf else \
        np.full(len(Ls), np.nan)
    # ---- SFF ----
    rho_bar = np.mean([r(E0) for r in rhos if r is not None]) if any(rhos) else np.nan
    tH = 2 * np.pi * rho_bar
    ts_fit = np.linspace(0, tau_fit * tH, nfit + 1)[1:]
    Kc = np.zeros(len(ts_fit))
    Kg = np.zeros(len(ts_fit))
    plat = 0.0
    n = 0
    for sh in shifts:
        for i, E in enumerate(used):
            if rhos[i] is None:
                continue
            k, pl = sff_one(E, ts_fit, E0 + sh, sig, rhos[i])
            Kc += k
            Kg += sff_goe(ts_fit, E0 + sh, sig, rhos[i], E.min(), E.max())
            plat += pl
            n += 1
    if n:
        Kc /= n
        Kg /= n
        plat /= n
    slope_goe = sig / np.sqrt(np.pi)
    f_raw = slope_through_origin(ts_fit, Kc) / slope_goe
    f_ref = slope_through_origin(ts_fit, Kg) / slope_goe
    out = dict(r_mean=float(np.nanmean(rr)), r_sem=float(np.nanstd(rr) / np.sqrt(max(len(rr), 1))),
               ru_mean=float(np.nanmean(ru)),
               ru_sem=float(np.nanstd(ru) / np.sqrt(max(len(ru), 1))),
               n_r=int(len(rr)), n_win=float(np.mean([len(w) for w in wins])),
               n_win_tot=int(sum(len(w) for w in wins)),
               sigma2=sig2, Ls=Ls, f=f_raw, f_ref=f_ref,
               f_cal=f_raw / f_ref if f_ref else np.nan,
               rho=rho_bar, tH=tH, plateau=plat)
    if ts_plot is not None:
        Kc2 = np.zeros(len(ts_plot))
        Kg2 = np.zeros(len(ts_plot))
        n2 = 0
        for sh in shifts:
            for i, E in enumerate(used):
                if rhos[i] is None:
                    continue
                k, pl = sff_one(E, ts_plot, E0 + sh, sig, rhos[i])
                Kc2 += k
                Kg2 += sff_goe(ts_plot, E0 + sh, sig, rhos[i], E.min(), E.max())
                n2 += 1
        out['Kc'] = Kc2 / max(n2, 1)
        out['Kgoe'] = Kg2 / max(n2, 1)
        out['ts'] = np.asarray(ts_plot)
    return out
