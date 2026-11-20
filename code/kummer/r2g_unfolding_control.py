"""Control for the size-scaled unfolding of the Letter's Figs. 3 and 4.

Backs the unfolding paragraph of the SM (Sec. "First-order matrix and spectral statistics"):

  "The degree of the polynomial grows with the size of the sub-block: for n levels it is n/100 rounded to the nearest
   integer, and at least 12. [...] at degree 12 the fit of the larger sub-blocks leaves a residual that varies slowly
   across the band, which raises the number variance at large L. As a control, we unfolded chiral GOE spectra with the
   shapes of the reference direction's sub-blocks in the same way. They reproduce the GOE number variance for L <= 10
   at E = 6, 7, 8 [...]. At E = 6 the sub-blocks have between 451 and 1366 levels, and the fit cannot be refined
   further without absorbing them."

The unfolding is the one of code/figure-reproduction/fig03_spacings_twocol.py: Chebyshev fit of the staircase
N(nu_i) = i,
5% trimmed at each end.  Chiral GOE spectra: singular values of a real Gaussian matrix with as many rows as twisted
states and as many columns as distinct values (bidiagonal model).  Three tables are printed and written to
r2_out/Unfolding_control.json:

  A. per sub-block shape and per FIXED degree (12, 20, 30, 50): Sigma^2(L) and the residual of the fitted staircase,
     for the data of the reference direction and for the chiral GOE.  Shows where over-fitting sets in (below about
     40 levels per fit parameter) and that degree 12 under-fits the data at E = 7, 8 (residual 2-8 levels against
     0.7-0.8 for the chiral GOE).
  B. per level, pooled as in Fig. 3, for the rules degree = 12 (the earlier choice) and degree = max(12, n / K) with
     K = 200, 100 (the Letter), 60: data against chiral GOE.
  C. r2_out/Unfolding_control_residuals_E8.png: fitted staircase minus staircase for the five E = 8 sub-blocks at
     degree 12 and at the Letter's degree.

Usage:  python r2g_unfolding_control.py [--samples=24] [--no-plot]        (about two minutes)
"""
import json
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import eigvalsh_tridiagonal

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
TRIM = 0.05
LC = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
GOE = [0.446, 0.584, 0.768, 0.909, 1.050]


def fit(x, deg):
    """Chebyshev coefficients of the staircase fit, the variable t in [-1, 1], and the distinct sorted values."""
    x = np.unique(np.round(np.asarray(x, dtype=float), 9))
    t = (2.0 * x - x[0] - x[-1]) / (x[-1] - x[0])
    c, (_, rank, _, _) = cheb.chebfit(t, np.arange(1, x.size + 1), deg, rcond=1e-15, full=True)
    if rank != deg + 1:
        raise np.linalg.LinAlgError('staircase fit of degree %d on %d levels is rank deficient (%d)' % (deg, x.size, rank))
    return c, t, x


def unfold(x, deg):
    """(unfolded levels on the retained range, sd of fit - staircase there, monotone)."""
    c, t, x = fit(x, deg)
    n = x.size
    xi = cheb.chebval(t, c)
    k = int(TRIM * n)
    tt = np.linspace(t[k], t[n - k - 1], 40000)
    return xi[k:n - k], float((xi - np.arange(1, n + 1))[k:n - k].std()), bool(np.all(cheb.chebval(tt, cheb.chebder(c)) > 0))


def number_variance(xi, Ls, npos=4000):
    out = []
    for L in Ls:
        a = np.linspace(xi[0], xi[-1] - L, npos)
        out.append(np.var(np.searchsorted(xi, a + L) - np.searchsorted(xi, a)))
    return np.array(out)


def chiral_goe(nt, n, rng):
    """Singular values of an nt x n real Gaussian matrix (bidiagonal model)."""
    d = np.sqrt(rng.chisquare(np.arange(nt, nt - n, -1)))
    e = np.sqrt(rng.chisquare(np.arange(n - 1, 0, -1)))
    return np.sqrt(np.clip(eigvalsh_tridiagonal(d ** 2 + np.concatenate([[0.0], e ** 2]), d[:-1] * e), 0, None))


def data():
    """{E: [(name, values, rows of the matching chiral GOE matrix), ...]} for the reference direction."""
    z = np.load(os.path.join(R2, 'Generic_firstorder_qgen_L7.npz'), allow_pickle=False)
    z8 = np.load(os.path.join(R2, 'Generic_probe_qgen_E8_mf_values.npz'), allow_pickle=False)
    h8 = np.load(os.path.join(R2, 'Hidden_firstorder_E8_values.npz'), allow_pickle=False)
    nt6 = {899: 1511, 451: 917, 1366: 2392}
    B = {6: [(k, z[k], nt6[np.unique(np.round(z[k], 9)).size]) for k in sorted(z.keys()) if k.startswith('E6_')],
         7: [(k, z[k], 7056) for k in sorted(z.keys()) if k.startswith('E7_')],
         8: [('E8_' + k, z8[k], 19856) for k in ('rho2_labpp', 'rho3_labpp', 'rho6_labpp')]
            + [('E8_eigenspace_6137', h8['qgen__leaf0'], 13289), ('E8_eigenspace_3079', h8['qgen__leaf1'], 6667)]}
    assert [np.unique(np.round(b, 9)).size for _, b, _ in B[6]] == [899, 451, 1366, 1366, 1366]
    assert [np.unique(np.round(b, 9)).size for _, b, _ in B[8]] == [9208, 9208, 9208, 6137, 3079]
    return B


def main(nsamp=24, plot=True):
    B = data()
    rng = np.random.default_rng(20261116)
    res = dict(L=LC.tolist(), GOE=GOE, A={}, B={})
    np.set_printoptions(linewidth=200)

    print('A. Fixed degree.  Sigma^2 at L = 1, 2, 5, 10, 20 (GOE %s), residual = sd of fitted staircase - staircase' % GOE)
    groups = [('E=8, three sub-blocks of 9208', [B[8][0], B[8][1], B[8][2]]), ('E=8, eigenspace 6137', [B[8][3]]),
              ('E=8, eigenspace 3079', [B[8][4]]), ('E=7, four sub-blocks of 3536', B[7]),
              ('E=6, three sub-blocks of 1366', B[6][2:]), ('E=6, eigenspace 899', [B[6][0]]), ('E=6, eigenspace 451', [B[6][1]])]
    for name, blocks in groups:
        n, nt = np.unique(np.round(blocks[0][1], 9)).size, blocks[0][2]
        res['A'][name] = {}
        print('\n%s (chiral GOE shape %d x %d)' % (name, nt, n))
        nul_spec = [chiral_goe(nt, n, rng) for _ in range(nsamp)]
        for deg in (12, 20, 30, 50):
            d = [unfold(b, deg) for _, b, _ in blocks]
            u = [unfold(s, deg) for s in nul_spec]
            ds = np.mean([number_variance(x[0], LC) for x in d], axis=0)
            us = np.array([number_variance(x[0], LC) for x in u])
            row = dict(levels_per_parameter=n / (deg + 1), data=ds.tolist(), data_residual=float(np.mean([x[1] for x in d])),
                       goe=us.mean(axis=0).tolist(), goe_sd=us.std(axis=0).tolist(), goe_residual=float(np.mean([x[1] for x in u])),
                       monotone=bool(all(x[2] for x in d)))
            res['A'][name][str(deg)] = row
            print('  degree %2d (%4.0f levels per parameter): data %s residual %5.2f | chiral GOE %s residual %4.2f'
                  % (deg, row['levels_per_parameter'], np.round(ds, 3), row['data_residual'], np.round(us.mean(axis=0), 3),
                     row['goe_residual']), flush=True)

    print('\nB. Pooled per level as in Fig. 3 (weights = numbers of retained levels)')
    for K in (None, 200, 100, 60):
        label = 'degree 12' if K is None else 'degree max(12, n/%d)' % K
        res['B'][label] = {}
        print('\nrule: %s%s' % (label, '   <- the letter' if K == 100 else ''))
        for E in (6, 7, 8):
            sizes = [np.unique(np.round(b, 9)).size for _, b, _ in B[E]]
            degs = [12 if K is None else max(12, int(round(n / K))) for n in sizes]
            try:
                d = [unfold(b, dg) for (_, b, _), dg in zip(B[E], degs)]
            except np.linalg.LinAlgError as err:          # happens only beyond the Letter's rule (K = 60 at E = 8)
                res['B'][label][str(E)] = dict(degrees=degs, skipped=str(err))
                print('  E=%d degrees %-20s skipped: %s' % (E, degs, err), flush=True)
                continue
            ds = np.average([number_variance(x[0], LC) for x in d], axis=0, weights=[x[0].size for x in d])
            s2 = float(np.mean(np.concatenate([np.diff(x[0]) / np.diff(x[0]).mean() for x in d]) ** 2))
            us = []
            for _ in range(max(8, nsamp // 2)):
                u = [unfold(chiral_goe(nt, n, rng), dg) for (_, _, nt), n, dg in zip(B[E], sizes, degs)]
                us.append(np.average([number_variance(x[0], LC) for x in u], axis=0, weights=[x[0].size for x in u]))
            us = np.array(us)
            res['B'][label][str(E)] = dict(degrees=degs, data=ds.tolist(), s2=s2, residuals=[x[1] for x in d],
                                           monotone=bool(all(x[2] for x in d)), goe=us.mean(axis=0).tolist(),
                                           goe_sd=us.std(axis=0).tolist())
            print('  E=%d degrees %-20s data %s <s^2> %.3f residuals %s | chiral GOE %s +- %.3f at L=10'
                  % (E, degs, np.round(ds, 3), s2, ' '.join('%.1f' % x[1] for x in d), np.round(us.mean(axis=0), 3), us.std(axis=0)[3]),
                  flush=True)

    with open(os.path.join(R2, 'Unfolding_control.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('\nwrote r2_out/Unfolding_control.json')

    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(5, 1, figsize=(7.5, 8.2))
        for ax, (name, b, nt) in zip(axs, B[8]):
            n = np.unique(np.round(b, 9)).size
            k = int(TRIM * n)
            for deg, col, lw in ((12, '#c0392b', 0.9), (max(12, int(round(n / 100))), '#1f6fd0', 0.7)):
                c, t, x = fit(b, deg)
                r = (cheb.chebval(t, c) - np.arange(1, n + 1))[k:n - k]
                ax.plot(np.arange(n)[k:n - k] / n, r, color=col, lw=lw, label='degree %d: sd %.1f' % (deg, r.std()))
            ax.axhline(0, color='k', lw=0.4)
            ax.set_ylabel('fit - staircase')
            ax.legend(loc='upper right', fontsize=7, ncol=2, frameon=False)
            ax.set_title('E = 8, reference direction, %s (%d levels)' % (name, n), fontsize=8, loc='left', pad=2)
            ax.tick_params(labelsize=7)
        axs[-1].set_xlabel('fraction of the levels (5% trimmed at each end)')
        fig.tight_layout()
        fig.savefig(os.path.join(R2, 'Unfolding_control_residuals_E8.png'), dpi=170)
        print('wrote r2_out/Unfolding_control_residuals_E8.png')


if __name__ == '__main__':
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    main(nsamp=int(opts.get('samples', 24)), plot='no-plot' not in opts)
