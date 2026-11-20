"""Route 2 Phase C, GATE 14: full sub-sector matrices + exact spectra at large th_max
(designed to run on the Fir cluster for th_max=8, Lambda=4).

Does NOT modify any existing r2c_*.py file: it only imports the validated
r2c_gate10.build_subsectors builder and r2c_gate12_run.LAM_GRID (the lambda grid used for
the exact-vs-first-order comparison), and saves two files:

  r2_out/PhaseC4_matrices_<mode>_L<th_max>.npz
      E0_k  (bare levels of sub-sector k, untwisted first then twisted)
      M_k   (the (nt, nu) coupling matrix of sub-sector k)
      meta  (one row per sub-sector: rho, dim, *label, n, nu -- same convention as
             r2c_gate10.main)

  r2_out/PhaseC4_spectra_<mode>_L<th_max>.npz   -- SAME LAYOUT as r2c_gate10.main:
      E0_k, ev_<lam>_k (exact H(lambda) eigenvalues only, no first-order comparison),
      meta, lambdas, th_max, t_build, t_eig

Usage:  python r2c_fir_run.py [th_max] [mode]     mode = prim | full   (default 8, prim)
"""
import os
import sys
import time

import numpy as np

import r2c_gate10 as g10
from r2c_gate12_run import LAM_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))


def main(th_max=8, mode='prim'):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    S, subs, names = g10.build_subsectors(th_max, mode)
    tb = time.time() - t0
    print('BUILD_TIME %.3f' % tb, flush=True)
    print('built %d fully resolved sub-sectors in %.1fs (extra Z_2: %s)'
          % (len(subs), tb, names), flush=True)

    meta = []
    mat_data = {}
    for k, d in enumerate(subs):
        ss = d['sub']
        meta.append((d['rho'], d['dim']) + tuple(d['label']) + (ss.n, ss.nu))
        mat_data['E0_%d' % k] = ss.E0
        mat_data['M_%d' % k] = ss.M
    meta_arr = np.array(meta, dtype=int)
    mat_data['meta'] = meta_arr
    mat_fn = os.path.join(OUT, 'PhaseC4_matrices_%s_L%d.npz' % (mode, th_max))
    np.savez_compressed(mat_fn, **mat_data)
    print('wrote', mat_fn, flush=True)

    data = {}
    for k, d in enumerate(subs):
        data['E0_%d' % k] = d['sub'].E0
    times = []
    lams = sorted(LAM_GRID)
    for lam in lams:
        t1 = time.time()
        for k, d in enumerate(subs):
            data['ev_%g_%d' % (lam, k)] = d['sub'].eig(lam)
        times.append(time.time() - t1)
        print('  lambda = %-6g diagonalised (%.1fs)' % (lam, times[-1]), flush=True)
    data['meta'] = meta_arr
    data['lambdas'] = np.array(lams)
    data['t_build'] = np.array([tb])
    data['t_eig'] = np.array(times)
    data['th_max'] = np.array([th_max])
    spec_fn = os.path.join(OUT, 'PhaseC4_spectra_%s_L%d.npz' % (mode, th_max))
    np.savez_compressed(spec_fn, **data)
    print('wrote', spec_fn, '(total %.1fs)' % (time.time() - t0), flush=True)

    print('TOTAL_TIME %.3f' % (time.time() - t0), flush=True)
    print('GATE14_RUN_DONE th_max=%d mode=%s' % (th_max, mode), flush=True)


if __name__ == '__main__':
    thm = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    md = sys.argv[2] if len(sys.argv) > 2 else 'prim'
    main(thm, md)
