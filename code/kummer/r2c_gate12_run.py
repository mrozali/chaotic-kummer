"""Route 2 Phase C, GATE 12 (T1/T2/T3): beyond the radius of degenerate perturbation
theory -- exact vs first-order spectra, perturbative-control measures, and kernel counting
at the top bare level, for the truncated Hamiltonian H(lambda) = H_0 + lambda D_0.

Does NOT modify any existing r2c_*.py file: it only imports r2c_gate10.build_subsectors
(the validated builder) and r2c_gate11.window / r2c_gate11.unlifted (the validated window
and unlifted-level definitions), and adds the first-order (degenerate-perturbation-theory)
spectrum plus the T2/T3 diagnostics on top.

Usage:  python r2c_gate12_run.py <th_max> [mode]        mode = prim | full  (default prim)
Writes  r2_out/PhaseC2_spectra_<mode>_L<th_max>.npz   -- T1: exact (ev_<lam>_<k>) and
                                                          first-order (ev1_<lam>_<k>) spectra
        r2_out/PhaseC2_t2_<mode>_L<th_max>.npz        -- T2: pooled perturbative measures vs lam
        r2_out/PhaseC2_t3_<mode>_L<th_max>.csv        -- T3: per-sub-sector kernel-counting table

For the record: SubSector.E0 has the first `nu` entries untwisted (bare level E0_u), the
remaining `nt` twisted (bare level E0_t); SubSector.M has shape (nt, nu), M[i, j] is the
matrix element between twisted state i and untwisted state j, and
    H(lambda) = diag(E0) + lambda * [[0, M^dagger], [M, 0]] .
The first-order (degenerate-perturbation-theory) coupling Md keeps only same-bare-level
entries: Md[i, j] = M[i, j] if E0_t[i] == E0_u[j] else 0.  Its eigenvalues are the exact
diagonalisation of every degenerate block separately, i.e. E_B +/- lambda sigma_B to all
orders WITHIN a block -- the "first-order" spectrum used throughout this gate.
"""
import csv
import os
import sys
import time

import numpy as np

import r2c_gate10 as g10
import r2c_gate11 as g11

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))

LAM_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
            0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.25,
            0.30, 0.40, 0.50]


# --------------------------------------------------------------------------- linear algebra
def md_from_M(E0, M, nu):
    """First-order (level-diagonal) part of M: keep M[i, j] only where the twisted bare
    level equals the untwisted bare level, zero elsewhere."""
    Eu = E0[:nu]
    Et = E0[nu:]
    same = (Et[:, None] == Eu[None, :])
    return np.where(same, M, 0.0)


def build_H1(E0, Md, nu, lam):
    n = len(E0)
    H = np.zeros((n, n), complex)
    H[np.arange(n), np.arange(n)] = E0
    if lam:
        H[nu:, :nu] = lam * Md
        H[:nu, nu:] = lam * Md.conj().T
    return H


def eig_first_order(E0, Md, nu, lam):
    return np.linalg.eigvalsh(build_H1(E0, Md, nu, lam))


def rank_tol(A, rtol=1e-9):
    """numpy.linalg.matrix_rank with an explicit RELATIVE tolerance (rtol * largest
    singular value), robust to A being empty in either dimension."""
    if A.size == 0:
        return 0
    sv = np.linalg.svd(A, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0:
        return 0
    return int(np.sum(sv > rtol * sv[0]))


# --------------------------------------------------------------------------------- T1 + T2
def run_t1_t2(th_max, mode, subs, Mds):
    """Exact + first-order spectra for the full lambda grid (T1), and, using the SAME
    build (M matrices still in memory), the perturbative-control measures of T2."""
    n_sub = len(subs)
    meta = []
    data = {}
    for k, d in enumerate(subs):
        ss = d['sub']
        meta.append((d['rho'], d['dim']) + tuple(d['label']) + (ss.n, ss.nu))
        data['E0_%d' % k] = ss.E0

    for lam in LAM_GRID:
        t1 = time.time()
        for k, d in enumerate(subs):
            ss = d['sub']
            data['ev_%g_%d' % (lam, k)] = ss.eig(lam)
            data['ev1_%g_%d' % (lam, k)] = eig_first_order(ss.E0, Mds[k], ss.nu, lam)
        print('  lambda = %-6g  exact + first-order diagonalised  (%.1fs)'
              % (lam, time.time() - t1), flush=True)

    data['meta'] = np.array(meta, dtype=int)
    data['lambdas'] = np.array(sorted(LAM_GRID))
    data['th_max'] = np.array([th_max])
    fn = os.path.join(OUT, 'PhaseC2_spectra_%s_L%d.npz' % (mode, th_max))
    np.savez_compressed(fn, **data)
    print('wrote', fn, flush=True)

    # ---- T2: perturbative-control measures, window from r2c_gate11.window ----------------
    E0c, half = g11.window(th_max)
    lo, hi = E0c - half, E0c + half
    width = 2.0 * half
    rows = []
    for lam in LAM_GRID:
        d1_all, d2_all = [], []
        dist_ratio_num, dist_n_tot = 0.0, 0
        raw_dist_num, raw_dist_den = 0.0, 0
        for k, d in enumerate(subs):
            ss = d['sub']
            E0, nu, M = ss.E0, ss.nu, ss.M
            Eu, Et = E0[:nu], E0[nu:]
            same = (Et[:, None] == Eu[None, :])
            absM2 = np.abs(M) ** 2
            d1_u = lam * np.sqrt(np.sum(np.where(same, absM2, 0.0), axis=0))
            d1_t = lam * np.sqrt(np.sum(np.where(same, absM2, 0.0), axis=1))
            diff_u = np.where(same, 1.0, np.abs(Eu[None, :] - Et[:, None]))
            diff_t = np.where(same, 1.0, np.abs(Et[:, None] - Eu[None, :]))
            d2_u = lam ** 2 * np.sum(np.where(~same, absM2, 0.0) / diff_u, axis=0)
            d2_t = lam ** 2 * np.sum(np.where(~same, absM2, 0.0) / diff_t, axis=1)
            win_u = (Eu >= lo) & (Eu <= hi)
            win_t = (Et >= lo) & (Et <= hi)
            d1_all.append(d1_u[win_u]); d1_all.append(d1_t[win_t])
            d2_all.append(d2_u[win_u]); d2_all.append(d2_t[win_t])
            # spectral-distance measure: per sub-sector local mean |exact - first-order|
            # divided by that sub-sector's own mean level spacing in the window, then
            # pooled level-weighted over sub-sectors.
            ev_ex = np.sort(data['ev_%g_%d' % (lam, k)])
            ev_fo = np.sort(data['ev1_%g_%d' % (lam, k)])
            inwin = (ev_fo >= lo) & (ev_fo <= hi)
            nk = int(np.sum(inwin))
            if nk > 0:
                mean_dist_k = float(np.mean(np.abs(ev_ex[inwin] - ev_fo[inwin])))
                spacing_k = width / nk
                dist_ratio_num += nk * (mean_dist_k / spacing_k)
                dist_n_tot += nk
                raw_dist_num += nk * mean_dist_k
                raw_dist_den += nk
        d1c = np.concatenate(d1_all) if d1_all else np.array([np.nan])
        d2c = np.concatenate(d2_all) if d2_all else np.array([np.nan])
        mean_d1 = float(np.mean(d1c))
        mean_d2 = float(np.mean(d2c))
        ratio = mean_d2 / mean_d1 if mean_d1 else np.nan
        dist_over_spacing = dist_ratio_num / dist_n_tot if dist_n_tot else np.nan
        raw_dist = raw_dist_num / raw_dist_den if raw_dist_den else np.nan
        rows.append(dict(lam=lam, mean_delta1=mean_d1, mean_delta2=mean_d2, ratio=ratio,
                          delta2_over_s=mean_d2 / 1.0, mean_abs_dist=raw_dist,
                          dist_over_spacing=dist_over_spacing, n_in_window=dist_n_tot,
                          n_delta_states=int(d1c.size)))
        print('  T2  lambda = %-6g  <D1>=%.4g  <D2>=%.4g  <D2>/<D1>=%.4g  '
              'dist/spacing=%.4g' % (lam, mean_d1, mean_d2, ratio, dist_over_spacing),
              flush=True)

    t2fn = os.path.join(OUT, 'PhaseC2_t2_%s_L%d.npz' % (mode, th_max))
    np.savez(t2fn,
              lam=np.array([r['lam'] for r in rows]),
              mean_delta1=np.array([r['mean_delta1'] for r in rows]),
              mean_delta2=np.array([r['mean_delta2'] for r in rows]),
              ratio=np.array([r['ratio'] for r in rows]),
              delta2_over_s=np.array([r['delta2_over_s'] for r in rows]),
              mean_abs_dist=np.array([r['mean_abs_dist'] for r in rows]),
              dist_over_spacing=np.array([r['dist_over_spacing'] for r in rows]),
              n_in_window=np.array([r['n_in_window'] for r in rows]),
              n_delta_states=np.array([r['n_delta_states'] for r in rows]))
    print('wrote', t2fn, flush=True)
    return data


# --------------------------------------------------------------------------------------- T3
def run_t3(th_max, mode, subs, Mds, data, lam3=0.3):
    top = th_max
    below = th_max - 1
    rows = []
    tot = dict(n_top_u=0, n_top_t=0, fo_zero=0, pred=0, obs=0, pred_below=0, obs_below=0)

    def pred_level(E0, M, Eu, Et, L):
        u_idx = np.where(Eu == L)[0]
        t_idx = np.where(Et == L)[0]
        r_u = rank_tol(M[:, u_idx]) if u_idx.size else 0
        r_t = rank_tol(M[t_idx, :]) if t_idx.size else 0
        return (u_idx.size - r_u) + (t_idx.size - r_t), u_idx, t_idx

    for k, d in enumerate(subs):
        ss = d['sub']
        E0, nu, M, Md = ss.E0, ss.nu, ss.M, Mds[k]
        Eu, Et = E0[:nu], E0[nu:]

        pred_top, u_top, t_top = pred_level(E0, M, Eu, Et, top)
        pred_below, u_bel, t_bel = pred_level(E0, M, Eu, Et, below)

        Bd = Md[np.ix_(t_top, u_top)] if (t_top.size and u_top.size) \
            else np.zeros((t_top.size, u_top.size))
        r_b = rank_tol(Bd)
        fo_zero = (u_top.size - r_b) + (t_top.size - r_b)

        ev_top = data['ev_%g_%d' % (lam3, k)]
        obs_top = int(np.sum(np.abs(ev_top - float(top)) < 1e-9))
        obs_below = int(np.sum(np.abs(ev_top - float(below)) < 1e-9))

        row = dict(rho=d['rho'], label=d['label'], n_top_u=int(u_top.size),
                   n_top_t=int(t_top.size), fo_zero=int(fo_zero), pred_top=int(pred_top),
                   obs_top=obs_top, pred_below=int(pred_below), obs_below=obs_below)
        rows.append(row)
        tot['n_top_u'] += row['n_top_u']; tot['n_top_t'] += row['n_top_t']
        tot['fo_zero'] += row['fo_zero']; tot['pred'] += row['pred_top']
        tot['obs'] += row['obs_top']; tot['pred_below'] += row['pred_below']
        tot['obs_below'] += row['obs_below']

    t3fn = os.path.join(OUT, 'PhaseC2_t3_%s_L%d.csv' % (mode, th_max))
    with open(t3fn, 'w', newline='') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['rho', 'label', 'n_top_u', 'n_top_t', 'fo_zero_modes',
                       'predicted_unlifted_top', 'observed_unlifted_top',
                       'predicted_unlifted_level%d' % below, 'observed_unlifted_level%d' % below])
        for row in rows:
            wcsv.writerow([row['rho'], str(row['label']).replace(',', ';'), row['n_top_u'],
                           row['n_top_t'], row['fo_zero'], row['pred_top'], row['obs_top'],
                           row['pred_below'], row['obs_below']])
        wcsv.writerow(['TOTAL', '', tot['n_top_u'], tot['n_top_t'], tot['fo_zero'], tot['pred'],
                       tot['obs'], tot['pred_below'], tot['obs_below']])
    print('wrote', t3fn, flush=True)

    all_match_top = all(r['pred_top'] == r['obs_top'] for r in rows)
    all_match_below = all(r['pred_below'] == r['obs_below'] for r in rows)
    print('T3: predicted == observed in EVERY sub-sector at top level %d : %s'
          % (top, all_match_top), flush=True)
    print('T3: predicted == observed in EVERY sub-sector at level %d : %s'
          % (below, all_match_below), flush=True)
    print('T3 totals: top level %d  predicted=%d observed=%d  |  level %d  predicted=%d '
          'observed=%d' % (top, tot['pred'], tot['obs'], below, tot['pred_below'],
                            tot['obs_below']), flush=True)
    return rows, tot, all_match_top, all_match_below


# --------------------------------------------------------------------------------------- main
def main(th_max=6, mode='prim'):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    S, subs, names = g10.build_subsectors(th_max, mode)
    tb = time.time() - t0
    print('BUILD_TIME %.3f' % tb, flush=True)
    print('built %d fully resolved sub-sectors (extra Z2: %s)' % (len(subs), names), flush=True)

    Mds = [md_from_M(d['sub'].E0, d['sub'].M, d['sub'].nu) for d in subs]

    data = run_t1_t2(th_max, mode, subs, Mds)
    run_t3(th_max, mode, subs, Mds, data)

    print('TOTAL_TIME %.3f' % (time.time() - t0), flush=True)
    print('GATE12_RUN_DONE th_max=%d mode=%s' % (th_max, mode), flush=True)


if __name__ == '__main__':
    thm = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    md = sys.argv[2] if len(sys.argv) > 2 else 'prim'
    main(thm, md)
