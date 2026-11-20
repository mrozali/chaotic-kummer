"""Route 2 Phase C, PART 2 (GATE 10): the truncated Hamiltonian H(lambda) = H_0 + lambda D_0
beyond the radius of degenerate perturbation theory.

Usage:  python r2c_gate10.py [th_max] [mode]     mode = prim | full
Writes  data/r2_out/PhaseC_spectra_<mode>_L<th_max>.npz  (all eigenvalues) and the CSVs
consumed by r2c_gate11.py (the report / figures driver).
"""
import os
import sys
import time
import numpy as np

import r2c_group as gr
import r2c_space as sp
import r2c_build as bd
import r2c_sym as sy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))

LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
LAM_FINE = [0.07, 0.12, 0.15, 0.17, 0.25, 0.4]


def build_subsectors(th_max, mode='prim', verbose=True):
    S = sp.Space(th_max, mode=mode, verbose=verbose)
    g = gr.G()
    subs = []
    for r in range(23):
        if not g.phys[r]:
            continue
        t0 = time.time()
        sec = bd.IrrepSector(S, r, verbose=False)
        if sec.n == 0:
            continue
        names, parts = sy.split_all(sec, S)
        for lab, ss in sorted(parts.items()):
            if ss.n:
                subs.append(dict(rho=r, dim=sec.d, label=lab, sub=ss))
        if verbose:
            print('  rho=%2d dim %d: n=%5d -> %d sub-sectors %s  (%.1fs)'
                  % (r, sec.d, sec.n, len(parts),
                     [parts[l].n for l in sorted(parts)], time.time() - t0), flush=True)
    return S, subs, names


def main(th_max=6, mode='prim', fine=False):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    S, subs, names = build_subsectors(th_max, mode)
    tb = time.time() - t0
    print('built %d fully resolved sub-sectors in %.1fs (extra Z_2: %s)'
          % (len(subs), tb, names), flush=True)
    data = {}
    meta = []
    for k, d in enumerate(subs):
        meta.append((d['rho'], d['dim']) + tuple(d['label']) + (d['sub'].n, d['sub'].nu))
        data['E0_%d' % k] = d['sub'].E0
    times = []
    lams = LAMBDAS + (LAM_FINE if fine else [])
    for lam in lams:
        t1 = time.time()
        for k, d in enumerate(subs):
            data['ev_%g_%d' % (lam, k)] = d['sub'].eig(lam)
        times.append(time.time() - t1)
        print('  lambda = %-6g diagonalised (%.1fs)' % (lam, times[-1]), flush=True)
    data['meta'] = np.array(meta, dtype=int)
    data['lambdas'] = np.array(sorted(lams))
    data['t_build'] = np.array([tb])
    data['t_eig'] = np.array(times)
    data['th_max'] = np.array([th_max])
    fn = os.path.join(OUT, 'PhaseC_spectra_%s_L%d.npz' % (mode, th_max))
    np.savez_compressed(fn, **data)
    print('wrote', fn, '(total %.1fs)' % (time.time() - t0))
    return fn


if __name__ == '__main__':
    thm = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    md = sys.argv[2] if len(sys.argv) > 2 else 'prim'
    fine = len(sys.argv) > 3 and sys.argv[3] == 'fine'
    main(thm, md, fine)
