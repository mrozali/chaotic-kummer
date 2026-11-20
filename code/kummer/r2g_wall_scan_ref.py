"""r2g_wall_scan_ref.py -- random-matrix references for the wall scan (r2g_wall_scan.py).

For every level the scan pools sub-blocks of one shape nt x n (twisted states x rank):
  E = 3  4 x (49 x 25)      E = 4  3 x (192 x 76)      E = 5  4 x (729 x 369)      E = 6  3 x (2392 x 1366)

1. GOE reference: pooled <r> of independent real Gaussian nt x n matrices (the Letter's chiral GOE reference);
   dense SVD at E <= 5, the Dumitriu-Edelman bidiagonal model at E = 6.
2. Crossover model for a sub-block split by a wall symmetry W.  Near a wall, q = cos(delta) q_w + sin(delta) n
   with q_w on the wall and n its unit normal, so M_q = cos(delta) M_{q_w} + sin(delta) M_n, where M_{q_w}
   commutes with W and M_n anticommutes with it.  The B_4 equivalences preserve ||M_q||_F, so both parts have
   the same norm.  Model: two halves, M = [[A, t B], [t C, D]], t = tan(delta), all entries i.i.d. N(0,1),
   A: nt1 x n1, D: nt2 x n2 (nt1 = ceil(nt/2), n1 = ceil(n/2)).  No free parameter.
   At t = 0 this is the superposition of two independent spectra.
   Pooled curves: coordinate walls split every pooled sub-block; diagonal walls split two of the four
   characters at odd E and one of the three pooled ones at even E, and pair the others (read off the scan).

Output: r2_out/Wall_scan_ref.json
Usage: python r2g_wall_scan_ref.py [levels e.g. 3,4,5,6] [--nsamp=E:N,...]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import lapack

import f2_common as f2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
SHAPES = {3: (4, 49, 25), 4: (3, 192, 76), 5: (4, 729, 369), 6: (3, 2392, 1366)}      # (blocks, nt, n)
N_GOE = {3: 4000, 4: 2000, 5: 1000, 6: 1000}
N_MODEL = {3: 1000, 4: 400, 5: 80, 6: 8}
DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]       # degrees


def sv_bidiag(rng, m, n):
    d = np.sqrt(rng.chisquare(np.arange(m, m - n, -1, dtype=float)))
    e = np.sqrt(rng.chisquare(np.arange(n - 1, 0, -1, dtype=float)))
    w, info = lapack.dsterf(d * d + np.concatenate(([0.0], e * e)), d[:-1] * e)
    assert info == 0
    return np.sqrt(np.sort(np.abs(w)))


def sv(M):
    return np.sort(np.linalg.svd(M, compute_uv=False))


def goe_ref(rng, E, nsamp):
    nb, nt, n = SHAPES[E]
    out = []
    for _ in range(nsamp):
        g = [f2.gap_ratios(sv_bidiag(rng, nt, n) if E >= 6 else sv(rng.standard_normal((nt, n)))) for _ in range(nb)]
        out.append(float(np.concatenate(g).mean()))
    return np.array(out)


def split_model(rng, E, t, nsamp):
    """<r> of one split sub-block (per sample), for coupling t."""
    _, nt, n = SHAPES[E]
    nt1, n1 = (nt + 1) // 2, (n + 1) // 2
    nt2, n2 = nt - nt1, n - n1
    out = []
    for _ in range(nsamp):
        M = np.empty((nt, n))
        M[:nt1, :n1] = rng.standard_normal((nt1, n1))
        M[nt1:, n1:] = rng.standard_normal((nt2, n2))
        M[:nt1, n1:] = t * rng.standard_normal((nt1, n2))
        M[nt1:, :n1] = t * rng.standard_normal((nt2, n1))
        out.append(float(f2.gap_ratios(sv(M)).mean()))
    return np.array(out)


def main(levels, nsamp_model):
    path = os.path.join(OUT, 'Wall_scan_ref.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for E in levels:
        t0 = time.time()
        rng = np.random.default_rng([20261105, E])
        g = goe_ref(rng, E, N_GOE[E])
        nb = SHAPES[E][0]
        curve = []
        for dl in DELTAS:
            s = split_model(rng, E, np.tan(np.radians(dl)), nsamp_model.get(E, N_MODEL[E]))
            curve.append(dict(delta=dl, mean=float(s.mean()), sd=float(s.std(ddof=1)), nsamp=int(len(s))))
            print('  E=%d delta %5.2f: split sub-block <r> = %.4f (sd %.4f, n %d)' % (E, dl, s.mean(), s.std(ddof=1), len(s)),
                  flush=True)
        n_split_diag = 2 if E % 2 else 1
        res[str(E)] = dict(shape=SHAPES[E], goe_mean=float(g.mean()), goe_sd=float(g.std(ddof=1)), goe_nsamp=int(len(g)),
                           goe_model='bidiagonal' if E >= 6 else 'dense', split_curve=curve,
                           pooled_coord=[c['mean'] for c in curve],
                           pooled_diag=[(n_split_diag * c['mean'] + (nb - n_split_diag) * float(g.mean())) / nb
                                        for c in curve],
                           n_split_diag=n_split_diag, deltas=DELTAS, seconds=time.time() - t0)
        print('E=%d: GOE reference %.4f +- %.4f (%d samples)  [%.0f s]' % (E, g.mean(), g.std(ddof=1), len(g),
                                                                       time.time() - t0), flush=True)
        with open(path, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    levels = [int(x) for x in args[0].split(',')] if args else [3, 4, 5, 6]
    ns = {int(k): int(v) for k, v in (p.split(':') for p in opts['nsamp'].split(','))} if 'nsamp' in opts else {}
    main(levels, ns)
