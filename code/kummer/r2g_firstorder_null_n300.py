"""r2g_firstorder_null_n300.py -- the matched chiral-GOE reference of the generic first-order gap ratio with 300
samples at every level E = 3..8, for the Letter.

Reference: real Gaussian matrices of the sub-block shapes nt x rank (the Laguerre orthogonal ensemble), all singular
values, gap ratios per block (f2_common.gap_ratios), pooled over the blocks of the level; one sample = one draw of
every block.  Shapes (the reference direction (qgen), the same along a second random direction (qgen2) and the
three scans):
  E = 3  49x25 (x4)                          E = 6  1511x899, 917x451, 2392x1366 (x3)
  E = 4  136x50, 72x26, 192x76 (x3)          E = 7  7056x3536 (x4)
  E = 5  729x369 (x4)                        E = 8  19856x9208 (x3), the three sub-blocks without a hidden Z2
Samples per level: the existing dense ones, completed to 300 by the bidiagonal model.
  dense, E = 3..6   200 per level, regenerated exactly as in r2g_firstorder_nullcheck.py qgen 6 (not included in
                    this repository) (default_rng(2026),
                    levels and blocks in the order of Generic_firstorder_qgen_L6.json, 4-decimal mean and sd logged in
                    generic_firstorder_qgen_L6_null.log; the regeneration is checked against them)
  dense, E = 7, 8   the 50 and 20 stored samples of Generic_firstorder_null_E78.json (seed 20261103)
  bidiagonal        the rest.  The singular values of an m x n (m >= n) real Gaussian matrix have the same distribution
                    as those of the n x n upper bidiagonal matrix B with independent diagonal chi_m, chi_{m-1}, ...,
                    chi_{m-n+1} and superdiagonal chi_{n-1}, ..., chi_1 (Householder bidiagonalisation keeps the
                    remaining submatrix Gaussian; Silverstein 1985, Dumitriu and Edelman 2002).  They are the square
                    roots of the eigenvalues of the tridiagonal B^T B (LAPACK dsterf); B is well conditioned
                    (sigma_max / sigma_min ~ 5 at E = 8), so they are accurate to ~1e-13 relative, far below the mean
                    spacing.  Sample k of level E uses default_rng([20261104, E, k]), whichever process computes it.
  The merge compares the dense and the bidiagonal samples level by level, and 'validate' compares the two on two
  further shapes with many samples.
Phases (each call stops after --budget seconds and resumes from the state file):
  python r2g_firstorder_null_n300.py dense    [--budget 150]   regenerate the dense E = 3..6 samples (4 BLAS threads)
  python r2g_firstorder_null_n300.py bidiag   [--budget 150]   bidiagonal samples up to 300 (4 worker processes)
  python r2g_firstorder_null_n300.py validate [--budget 150]   dense vs bidiagonal, 300x150 (3000 each), 917x451 (1000)
  python r2g_firstorder_null_n300.py merge                     write r2_out/Generic_firstorder_null_n300.json
  python r2g_firstorder_null_n300.py status
State (raw samples, resumable): Generic_firstorder_null_n300_state.json in the record folder, or the file given by
the environment variable N300_STATE.
The merged file keeps every sample.
"""
import json
import os
import sys
import time

PHASE = sys.argv[1] if len(sys.argv) > 1 else 'status'
BUDGET = float(sys.argv[sys.argv.index('--budget') + 1]) if '--budget' in sys.argv else 150.0
NWORK = 4
_threads = '4' if PHASE in ('dense', 'validate') else '1'
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = _threads

import numpy as np                                                       # noqa: E402
from scipy.linalg import lapack                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f2_common as f2                                                   # noqa: E402

R2 = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
STATE = os.path.expanduser(os.environ.get('N300_STATE') or os.path.join(R2, 'Generic_firstorder_null_n300_state.json'))
OUT = os.path.join(R2, 'Generic_firstorder_null_n300.json')
TARGET = 300
SEED_BIDIAG = 20261104
LOG_L6 = {3: (0.5345, 0.0314), 4: (0.5296, 0.0174), 5: (0.5311, 0.0088), 6: (0.5309, 0.0041)}   # mean, sd (ddof 0)

_J = json.load(open(os.path.join(R2, 'Generic_firstorder_qgen_L6.json')))
LEVELS_L6 = [int(E) for E in _J['levels']]
SHAPES = {}
for _E in LEVELS_L6:
    _rows = [r for r in _J['leaves'][str(_E)]['rows'] if r['distinct'] >= 20]
    if _rows:
        SHAPES[_E] = [(int(r['nt']), int(r['rank'])) for r in _rows]
E78 = json.load(open(os.path.join(R2, 'Generic_firstorder_null_E78.json')))['levels']
for _E in (7, 8):
    SHAPES[_E] = [tuple(int(x) for x in s) for s in E78[str(_E)]['shapes']]
assert SHAPES[3] == [(49, 25)] * 4 and SHAPES[5] == [(729, 369)] * 4
assert SHAPES[6] == [(1511, 899), (917, 451)] + [(2392, 1366)] * 3
assert SHAPES[7] == [(7056, 3536)] * 4 and SHAPES[8] == [(19856, 9208)] * 3


def pooled_mean(sorted_vals):
    g = np.concatenate([f2.gap_ratios(v) for v in sorted_vals if len(v) >= 20])
    return float(g.mean()), int(g.size)


def sv_bidiag(rng, m, n):
    """Singular values of the bidiagonal model of an m x n real Gaussian matrix, ascending."""
    d = np.sqrt(rng.chisquare(np.arange(m, m - n, -1, dtype=float)))
    e = np.sqrt(rng.chisquare(np.arange(n - 1, 0, -1, dtype=float)))
    w, info = lapack.dsterf(d * d + np.concatenate(([0.0], e * e)), d[:-1] * e)
    assert info == 0
    return np.sqrt(np.sort(np.abs(w)))


def sv_dense(rng, m, n):
    return np.sort(np.linalg.svd(rng.standard_normal((m, n)), compute_uv=False))


def n_dense(E):
    return 200 if E <= 6 else int(E78[str(E)]['nsamp'])


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return dict(dense=dict(done=False, level_pos=0, k=0, rng=None, samples={}), bidiag={}, validate={})


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + '.tmp', 'w') as fh:
        json.dump(st, fh)
    os.replace(STATE + '.tmp', STATE)


def phase_dense(st):
    d = st['dense']
    rng = np.random.default_rng(2026)
    if d['rng'] is not None:
        rng.bit_generator.state = d['rng']
    levels = [E for E in LEVELS_L6 if E in SHAPES and E <= 6]
    t0 = time.time()
    while not d['done']:
        E = levels[d['level_pos']]
        sims = [sv_dense(rng, m, n) for m, n in SHAPES[E]]
        d['samples'].setdefault(str(E), []).append(pooled_mean(sims)[0])
        d['k'] += 1
        if d['k'] == 200:
            d['level_pos'] += 1
            d['k'] = 0
            d['done'] = d['level_pos'] == len(levels)
        d['rng'] = rng.bit_generator.state
        save_state(st)
        if time.time() - t0 > BUDGET:
            break
    print('dense:', {E: len(v) for E, v in d['samples'].items()}, 'done' if d['done'] else 'to be continued',
          '[%.0f s]' % (time.time() - t0))


def bidiag_sample(task):
    E, k = task
    rng = np.random.default_rng([SEED_BIDIAG, E, k])
    return E, k, pooled_mean([sv_bidiag(rng, m, n) for m, n in SHAPES[E]])[0]


def phase_bidiag(st):
    import multiprocessing as mp
    b = st['bidiag']
    tasks = [(E, k) for E in (8, 7, 6, 5, 4, 3) for k in range(n_dense(E), TARGET) if '%d:%d' % (E, k) not in b]
    t0, tsave = time.time(), time.time()
    if tasks:
        with mp.get_context('fork').Pool(NWORK) as pool:
            for E, k, v in pool.imap_unordered(bidiag_sample, tasks, chunksize=1):
                b['%d:%d' % (E, k)] = v
                if time.time() - tsave > 10:
                    save_state(st)
                    tsave = time.time()
                if time.time() - t0 > BUDGET:
                    break
    save_state(st)
    left = sum(1 for E in range(3, 9) for k in range(n_dense(E), TARGET) if '%d:%d' % (E, k) not in b)
    print('bidiag: %d done, %d left [%.0f s]' % (len(b), left, time.time() - t0))


def phase_validate(st):
    t0 = time.time()
    for (m, n), ns in (((300, 150), 3000), ((917, 451), 1000)):
        key = '%dx%d' % (m, n)
        if key in st['validate']:
            continue
        rng = np.random.default_rng([SEED_BIDIAG, 99, m, n])
        dv = np.array([pooled_mean([sv_dense(rng, m, n)])[0] for _ in range(ns)])
        bv = np.array([pooled_mean([sv_bidiag(rng, m, n)])[0] for _ in range(ns)])
        z = (bv.mean() - dv.mean()) / np.sqrt(dv.var(ddof=1) / ns + bv.var(ddof=1) / ns)
        st['validate'][key] = dict(nsamp_each=ns, dense_mean=dv.mean(), dense_sd=dv.std(ddof=1), bidiag_mean=bv.mean(),
                                   bidiag_sd=bv.std(ddof=1), z_mean=z, sd_ratio=bv.std(ddof=1) / dv.std(ddof=1),
                                   sd_rel_precision=1 / np.sqrt(2 * (ns - 1)))
        save_state(st)
        print('validate %s: dense %.5f +- %.5f, bidiagonal %.5f +- %.5f, means %.2f sigma apart, sd ratio %.3f'
              % (key, dv.mean(), dv.std(ddof=1), bv.mean(), bv.std(ddof=1), z, bv.std(ddof=1) / dv.std(ddof=1)))
        if time.time() - t0 > BUDGET:
            break


def phase_merge(st):
    d, b = st['dense'], st['bidiag']
    assert d['done'], 'dense phase incomplete'
    out = dict(doc=__doc__, target=TARGET, seed_bidiag=SEED_BIDIAG, levels={}, validation=st['validate'])
    for E in range(3, 9):
        dense = [float(x) for x in (d['samples'][str(E)] if E <= 6 else E78[str(E)]['samples'])]
        bid = [b['%d:%d' % (E, k)] for k in range(n_dense(E), TARGET)]
        a, dv, bv = np.array(dense + bid), np.array(dense), np.array(bid)
        lev = dict(shapes=SHAPES[E], n_gaps=int(sum(n - 2 for _, n in SHAPES[E])), nsamp=int(a.size),
                   n_dense=len(dense), n_bidiag=len(bid), mean=float(a.mean()), sd=float(a.std(ddof=1)),
                   se=float(a.std(ddof=1) / np.sqrt(a.size)), dense_mean=float(dv.mean()),
                   dense_sd=float(dv.std(ddof=1)), bidiag_mean=float(bv.mean()), bidiag_sd=float(bv.std(ddof=1)),
                   z_dense_vs_bidiag=float((bv.mean() - dv.mean()) / np.sqrt(dv.var(ddof=1) / dv.size
                                                                              + bv.var(ddof=1) / bv.size)),
                   dense_source=('regenerated, r2g_firstorder_nullcheck.py qgen 6, default_rng(2026)' if E <= 6
                                 else 'Generic_firstorder_null_E78.json, seed 20261103'),
                   samples_dense=dense, samples_bidiag=bid)
        if E <= 6:
            lm, ls = LOG_L6[E]
            lev['dense_vs_log'] = dict(log_mean=lm, log_sd_ddof0=ls, mean=float(dv.mean()), sd_ddof0=float(dv.std()),
                                       ok=bool(abs(dv.mean() - lm) <= 5.01e-5 and abs(dv.std() - ls) <= 5.01e-5))
        out['levels'][str(E)] = lev
        print('E=%d: %d dense + %d bidiagonal: mean %.4f sd %.4f | dense %.4f +- %.4f, bidiagonal %.4f +- %.4f '
              '(%.2f sigma)%s' % (E, len(dense), len(bid), a.mean(), a.std(ddof=1), dv.mean(), dv.std(ddof=1),
                                  bv.mean(), bv.std(ddof=1), lev['z_dense_vs_bidiag'],
                                  ('  log check ' + ('ok' if lev['dense_vs_log']['ok'] else 'MISMATCH')) if E <= 6 else ''))
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    st = load_state()
    if PHASE == 'dense':
        phase_dense(st)
    elif PHASE == 'bidiag':
        phase_bidiag(st)
    elif PHASE == 'validate':
        phase_validate(st)
    elif PHASE == 'merge':
        phase_merge(st)
    else:
        print('dense:', {E: len(v) for E, v in st['dense']['samples'].items()}, st['dense']['done'],
              '| bidiag:', len(st['bidiag']), '| validate:', list(st['validate']))
