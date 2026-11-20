"""r2g_firstorder_null_n3000.py -- the chiral-GOE reference of the generic first-order gap ratio with a few thousand
samples per level, so that the standard deviation sigma of the Letter's table is known to about one
per cent (with the 300 samples of r2g_firstorder_null_n300.py its own sampling error is 4 per cent).

Same reference and same estimator as r2g_firstorder_null_n300.py: one sample = one real Gaussian matrix nt x rank
per sub-block, all singular values, gap ratios per block (f2_common.gap_ratios), pooled over the blocks of the
level.  All samples from the bidiagonal model (Dumitriu and Edelman 2002; validated against dense matrices there).
Shapes: E = 3..7 from Generic_firstorder_null_n300.json.  E = 8: the three sub-blocks 19856 x 9208 and, with
--e8-leaves=n1,n2, the two eigenspaces of the hidden Z_2 of the trivial character found by
r2g_hidden_firstorder.py (n1 + n2 = 9216 values); their 19956 twisted states are divided in proportion to n1 : n2
(the first-order matrices do not fix the hidden-Z_2 charge of the twisted states that never couple, and sigma does
not depend on the number of rows at this accuracy: see 'rows_check' in the output).
Sample k of level E uses default_rng([20261116, E, k]).

Usage: python r2g_firstorder_null_n3000.py [--n=3000] [--levels=3,4,5,6,7,8] [--e8-leaves=n1,n2] [--workers=8]
Output: r2_out/Generic_firstorder_null_n3000.json (levels already present are kept unless recomputed)
"""
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np                                                       # noqa: E402
from scipy.linalg import lapack                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f2_common as f2                                                   # noqa: E402

R2 = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
OUT = os.path.join(R2, 'Generic_firstorder_null_n3000.json')
SEED = 20261116
SHAPES = {}


def sv_bidiag(rng, m, n):
    """Singular values of the bidiagonal model of an m x n real Gaussian matrix, ascending."""
    d = np.sqrt(rng.chisquare(np.arange(m, m - n, -1, dtype=float)))
    e = np.sqrt(rng.chisquare(np.arange(n - 1, 0, -1, dtype=float)))
    w, info = lapack.dsterf(d * d + np.concatenate(([0.0], e * e)), d[:-1] * e)
    assert info == 0
    return np.sqrt(np.sort(np.abs(w)))


def sample(task):
    tag, E, k, shapes = task
    rng = np.random.default_rng([SEED, E, k] + ([] if tag == 'main' else [99]))
    g = np.concatenate([f2.gap_ratios(sv_bidiag(rng, m, n)) for m, n in shapes])
    return tag, E, k, float(g.mean())


def summarise(v, shapes):
    v = np.asarray(v)
    return dict(shapes=[list(s) for s in shapes], n_gaps=int(sum(n - 2 for _, n in shapes)), nsamp=int(v.size),
                mean=float(v.mean()), sd=float(v.std(ddof=1)), se_mean=float(v.std(ddof=1) / np.sqrt(v.size)),
                se_sd=float(v.std(ddof=1) / np.sqrt(2 * (v.size - 1))), samples=[float(x) for x in v])


if __name__ == '__main__':
    import multiprocessing as mp
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    n = int(opts.get('n', 3000))
    levels = [int(x) for x in opts.get('levels', '3,4,5,6,7').split(',')]
    old = json.load(open(os.path.join(R2, 'Generic_firstorder_null_n300.json')))['levels']
    for E in range(3, 8):
        SHAPES[E] = [tuple(s) for s in old[str(E)]['shapes']]
    rows_check = None
    if 8 in levels:
        n1, n2 = [int(x) for x in opts['e8-leaves'].split(',')]
        assert n1 + n2 == 9216, (n1, n2)
        t1 = int(round(19956 * n1 / 9216))
        SHAPES[8] = [(19856, 9208)] * 3 + [(t1, n1), (19956 - t1, n2)]
        rows_check = [(19856, 9208)] * 3 + [(n1 + (19956 - 9216) // 2, n1), (19956 - n1 - (19956 - 9216) // 2, n2)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else dict(doc=__doc__, seed=SEED, levels={})
    tasks = [('main', E, k, SHAPES[E]) for E in levels for k in range(n)]
    if rows_check:                         # the same level with the twisted states divided equally beyond the rank
        tasks += [('rows', 8, k, rows_check) for k in range(min(n, 1000))]
    got = {}
    with mp.get_context('fork').Pool(int(opts.get('workers', 8))) as pool:
        for tag, E, k, v in pool.imap_unordered(sample, tasks, chunksize=4):
            got.setdefault((tag, E), {})[k] = v
    for E in levels:
        v = [got[('main', E)][k] for k in range(n)]
        res['levels'][str(E)] = summarise(v, SHAPES[E])
        r = res['levels'][str(E)]
        print('E=%d: %d samples, shapes %s: mean %.5f +- %.5f, sd %.5f +- %.5f   (300-sample record: sd %s)'
              % (E, n, sorted(set(SHAPES[E])), r['mean'], r['se_mean'], r['sd'], r['se_sd'],
                 ('%.5f' % old[str(E)]['sd']) if str(E) in old else 'n/a'), flush=True)
    if rows_check:
        v = [got[('rows', 8)][k] for k in sorted(got[('rows', 8)])]
        res['rows_check'] = summarise(v, rows_check)
        print('rows check at E=8, leaves %s: sd %.5f +- %.5f' % (rows_check[3:], res['rows_check']['sd'], res['rows_check']['se_sd']))
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=1)
