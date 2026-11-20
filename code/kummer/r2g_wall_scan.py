"""r2g_wall_scan.py -- unbiased direction scan: the resolved first-order gap ratio at E = 3..6 against the
angular distance of the direction from the nearest of the 16 walls.  The walls are the mirrors of the Letter.

Directions: qrand<seed> for a contiguous block of seeds (gp.cmat: q = N(0,1)^4 normalised, i.e. uniform on
S^3, with no selection), plus the reference direction (qgen) and a second random direction (qgen2), which are reported
separately.

Method: the lean first-order probe r2g_generic_probe_mf_lean.py with the direction dependence pulled out.
Off the walls the stabiliser H_c, its characters and the right-only Z_2's (Z_R, F_R) are the same for every
direction (checked for every direction), so the untwisted (r, label) bases Q do not depend on the direction,
and M_c = sum_mu q_mu M_mu is linear in q, c = q_0 + i (q_1 sigma_x + q_2 sigma_y + q_3 sigma_z).  Per level
and (r, label): Q exactly as in the lean probe (same seeds, CholQR3), then Z_mu = M_mu Q for the four
quaternion units.  Per direction: the singular values of sum_mu q_mu Z_mu = M_c Q, which are those of the
(r, label) sub-block, with the rank threshold 1e-9 gmax(q) and the distinct values rounded to 1e-9 as in the
lean probe; gap ratios on sub-blocks with >= 20 distinct values.

Pooling: odd E all four characters; even E the three non-trivial characters, because the trivial-character
sub-block carries the hidden Z_2 of the even levels, which this scan does not resolve (the Letter's rule for
the generic directions at E = 8).

Output: r2_out/Wall_scan_E<th>.json        per direction: q, angles to the 16 walls, per-sub-block rows, pooled <r>
        r2_out/Wall_scan_E<th>_values.npz  distinct values per direction and sub-block
Usage:  python r2g_wall_scan.py <th> [--seeds=101-148] [--extra=qgen,qgen2] [--chunk=400] [--with-trivial]
                                     [--direct-stab]   (stabiliser by the direct test instead of Y c Y^T = +-c)
                                     [--qfile=r2_out/Wall_scan_targets.json --tag=_targeted]
        --qfile adds directions given as unit quaternions (a JSON list of {"name", "q"}; written by
        r2g_wall_scan_targets.py); --tag is appended to the output names; --seeds= --extra= (empty) drop the
        default directions.
Runs of 2026-11-05 (2 cores): th = 5, 4, 6 with the defaults (73 s, 12 s, 24 min), th = 3 with --direct-stab
(7 min; all 50 directions have the generic stabiliser by the direct test).  Checks: Z_R commutes and F_R
anticommutes with each quaternion unit, M_c = sum q_mu M_mu to 2e-16, and the pooled <r> at E = 3, 5 of qrand101-116,
qgen and qgen2 reproduce the leaves of the truncated-Hamiltonian runs (Generic_firstorder_<dir>_L6.json) to 2e-10.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sl

import f2_common as f2
import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_L6 as pl6
import r2g_generic_probe_mf as mf
import r2g_generic_probe_mf_lean as lean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_POI = 0.5307, 0.3863

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
QB = [np.eye(2, dtype=complex), 1j * SX, 1j * SY, 1j * SZ]       # c = sum_mu q_mu QB_mu, as in gp.cmat


def c_of(q):
    return sum(q[m] * QB[m] for m in range(4))


def quat(c):
    """Unit quaternion of a direction (inverse of c_of on |q| = 1, ||c||_F = sqrt 2)."""
    q = np.array([np.trace(c).real / 2, ((c[0, 1] + c[1, 0]) / 2j).real, ((c[0, 1] - c[1, 0]) / 2).real,
                  ((c[0, 0] - c[1, 1]) / 2j).real])
    return q / np.linalg.norm(q)


# the 16 walls: q_mu = 0 and q_mu = +-q_nu.  Ten are mirrors of symmetries of the deformed theory (q_2 = 0,
# q_mu = 0 and q_mu = +-q_nu with mu, nu in {0,1,3}); the six q_2 = +-q_mu are exact on the invariant sector only.
E4 = np.eye(4)
WALLS = [('q%d=0' % i, E4[i], 'full') for i in range(4)]
WALLS += [('q%d=%sq%d' % (i, '+' if s > 0 else '-', j), (E4[i] - s * E4[j]) / np.sqrt(2),
           's0' if 2 in (i, j) else 'full') for i, j in itertools.combinations(range(4), 2) for s in (1, -1)]


def wall_angles(q):
    """Angles (degrees) between q and the 16 wall hyperplanes."""
    return np.array([np.degrees(np.arcsin(min(1.0, abs(float(n @ q))))) for _, n, _ in WALLS])


def run(th, dirs, chunk=400, with_trivial=False, direct_stab=False, qdirs=(), tag=''):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    T0 = time.time()
    log('# Wall scan at E = %d: %d directions  (%s)' % (th, len(dirs), time.strftime('%Y-%m-%d %H:%M')))
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built in %.1f s' % (th, time.time() - T0))

    # reference symmetry data from the reference direction (qgen); every scanned direction is checked against it below
    c_ref = gp.cmat('qgen')
    Hc, sgn, rule = gp.stabiliser(S, c_ref, 3)
    assert rule['Yb c Yb^T'] == len(Hc), rule           # the analytic rule c -> Y c Y^T reproduces H_c and signs
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    nH = len(Hc)
    log('reference stabiliser order %d, irreps %d, usable %s' % (nH, len(dims), irr))
    Mref = mf.MFree(S, th, c_ref, chunk=chunk)
    dt, du = Mref.shape
    nW = S.nW[th]
    Mmu = [mf.MFree(S, th, QB[m], chunk=chunk) for m in range(4)]
    act = lean.LeanAction(S, th)
    Ut = {gi: S.Utw[th][gi] for gi in Hc}
    log('M %d x %d (twisted x untwisted)' % (dt, du))

    # extra Z_2's exactly as in the lean probe (same rng stream), then the same test on each quaternion unit
    rng = np.random.default_rng(11)
    syms = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xt_mat = S.Utw[th][gi]
        x = rng.standard_normal((du, 3)) + 1j * rng.standard_normal((du, 3))
        rx = act.apply(gi, x, right_only=True)
        for lbl, M in [('c_ref', Mref)] + [('QB%d' % m, Mmu[m]) for m in range(4)]:
            lhs = pl6.apply_tw_right(Xt_mat, M @ x, nW)
            rhs = M @ rx
            nrm = max(float(np.max(np.abs(rhs))), 1e-30)
            cp = float(np.max(np.abs(lhs - rhs))) / nrm
            cm = float(np.max(np.abs(lhs + rhs))) / nrm
            s_ = +1 if cp < 1e-9 else (-1 if cm < 1e-9 else 0)
            if lbl == 'c_ref':
                if s_:
                    syms[nm] = (gi, Xt_mat, s_)
                sref = s_
            else:
                assert s_ == sref, (nm, lbl, cp, cm)
        log('    %s: %s with M_c and with each quaternion unit' % (nm, {1: 'commutes', -1: 'anticommutes', 0: 'no symmetry'}[sref]))
    names = sorted(syms)
    x = rng.standard_normal((du, 2)) + 1j * rng.standard_normal((du, 2))
    Mx = [Mmu[m] @ x for m in range(4)]
    qref = quat(c_ref)
    lin = float(np.max(np.abs(Mref @ x - sum(qref[m] * Mx[m] for m in range(4))))) / float(np.max(np.abs(Mref @ x)))
    log('    linearity M_c = sum q_mu M_mu at direction 1: rel. deviation %.1e' % lin)
    assert lin < 1e-12, lin

    # directions: quaternion, stabiliser check, wall angles, rank threshold
    D = []
    for name, qgiven in [(n, None) for n in dirs] + [(n, qq) for n, qq in qdirs]:
        if qgiven is None:
            c = gp.cmat(name)
        else:
            qq = np.asarray(qgiven, float)
            c = c_of(qq / np.linalg.norm(qq))
        q = quat(c)
        assert np.allclose(c_of(q), c, atol=1e-12), name
        if direct_stab:                                  # direct test of all 384 elements on the level-3 block
            Hq, sq, _ = gp.stabiliser(S, c, 3)
        else:                                            # analytic rule Y c Y^T = +-c (validated on the reference direction)
            Hq, sq = [], {}
            for gi in range(g.n):
                Yb = g.Yb[gi]
                cc = Yb @ c @ Yb.T
                for s_ in (1, -1):
                    if np.allclose(cc, s_ * c, atol=1e-9):
                        Hq.append(gi)
                        sq[gi] = s_
        generic = (Hq == Hc) and all(sq[k] == sgn[k] for k in Hc)
        ang = wall_angles(q)
        gm = float(np.max(np.abs(sum(q[m] * Mx[m] for m in range(4))))) / np.sqrt(du)
        D.append(dict(name=name, q=q.tolist(), generic=bool(generic), H_order=len(Hq), wall_angles=ang.tolist(),
                      nearest=WALLS[int(np.argmin(ang))][0], nearest_kind=WALLS[int(np.argmin(ang))][2],
                      delta=float(ang.min()), delta_full=float(min(a for a, w in zip(ang, WALLS) if w[2] == 'full')),
                      gmax=gm, rows=[]))
        if not generic:
            log('    WARNING %s: stabiliser order %d differs from the generic one' % (name, len(Hq)))
    log('    %d directions, %d with the generic stabiliser [%.1f s]' % (len(D), sum(d['generic'] for d in D),
                                                                       time.time() - T0))

    def weights(r):
        return [dims[r] / nH * np.conj(chars[r, elem_class[k]]) for k in range(nH)]

    tr_cache = {}

    def tr_u(gi, hs):
        if (gi, hs) not in tr_cache:
            tr_cache[(gi, hs)] = act.trace_with_right(gi, list(hs))
        return tr_cache[(gi, hs)]

    def dims_lab(r, lab):
        w = weights(r)
        tu = tt = 0.0 + 0.0j
        for k, gi in enumerate(Hc):
            if abs(w[k]) < 1e-15:
                continue
            trUt = np.trace(Ut[gi])
            for n_sub in range(len(names) + 1):
                for sub in itertools.combinations(range(len(names)), n_sub):
                    cu = np.prod([lab[i] for i in sub]) / 2 ** len(names)
                    ct = np.prod([lab[i] * syms[names[i]][2] for i in sub]) / 2 ** len(names)
                    hs = tuple(syms[names[i]][0] for i in sub)
                    tu += w[k] * cu * tr_u(gi, hs)
                    B = Ut[gi]
                    for i in sub:
                        B = B @ syms[names[i]][1]
                    tt += w[k] * sgn[gi] * ct * trUt * np.trace(B)
        nu, nt = int(np.rint(tu.real)), int(np.rint(tt.real))
        assert abs(tu - nu) < 1e-6 and abs(tt - nt) < 1e-6, (r, lab, tu, tt)
        return nu, nt

    def proj_lab(r, lab, X):
        Y = X
        for i, nm in enumerate(names):
            Y = 0.5 * (Y + lab[i] * act.apply(syms[nm][0], Y, right_only=True))
        acc = np.zeros_like(Y)
        for k, wk in enumerate(weights(r)):
            if abs(wk) < 1e-15:
                continue
            act.apply(Hc[k], Y, out=acc, weight=wk)
        return acc

    values = {}
    subs = []
    labs = list(itertools.product((+1, -1), repeat=len(names)))
    for r in irr:
        for lab in labs:
            t2 = time.time()
            nu, nt = dims_lab(r, lab)
            key = lean.labkey(r, lab)
            if nu == 0 or nt == 0:
                log('    irrep %d %s: nu %d nt %d, no coupling' % (r, list(lab), nu, nt))
                continue
            if th % 2 == 0 and r == 0 and not with_trivial:
                log('    irrep %d %s: nu %d nt %d, trivial character at even E (hidden Z_2): skipped' % (r, list(lab), nu, nt))
                subs.append(dict(key=key, rho=int(r), lab=list(lab), nu=nu, nt=nt, pooled=False, computed=False))
                continue
            rs = np.random.default_rng(1000 * r + 17 * (1 + sum((1 + l) // 2 * 2 ** i for i, l in enumerate(lab))))
            Q = np.empty((du, nu), complex)
            for k0 in range(0, nu, chunk):
                k1 = min(nu, k0 + chunk)
                X = rs.standard_normal((du, k1 - k0)) + 1j * rs.standard_normal((du, k1 - k0))
                Q[:, k0:k1] = proj_lab(r, lab, X)
            err = lean.cholqr_inplace(Q)
            assert err < 1e-11, err
            chk = proj_lab(r, lab, np.ascontiguousarray(Q[:, :4]))
            pdev = float(np.max(np.abs(chk - Q[:, :4])))
            assert pdev < 1e-8, pdev
            t3 = time.time()
            Z = []
            for m in range(4):
                Zm = np.empty((dt, nu), complex, order='F')
                for k0 in range(0, nu, chunk):
                    k1 = min(nu, k0 + chunk)
                    Zm[:, k0:k1] = Mmu[m].matvec(np.ascontiguousarray(Q[:, k0:k1]))
                Z.append(Zm)
            del Q
            t4 = time.time()
            pooled = not (th % 2 == 0 and r == 0)
            subs.append(dict(key=key, rho=int(r), lab=list(lab), nu=nu, nt=nt, pooled=pooled, computed=True))
            tsv = 0.0
            for d in D:
                q = d['q']
                A = np.asfortranarray(q[0] * Z[0] + q[1] * Z[1] + q[2] * Z[2] + q[3] * Z[3])
                t5 = time.time()
                s = sl.svd(A, compute_uv=False, overwrite_a=True, check_finite=False)
                tsv += time.time() - t5
                del A
                nz = s > 1e-9 * d['gmax']
                rank = int(nz.sum())
                dist = np.unique(np.round(s[nz], 9))
                gr = f2.gap_ratios(dist) if len(dist) >= 20 else np.zeros(0)
                values['%s__%s' % (d['name'], key)] = dist
                d['rows'].append(dict(key=key, rho=int(r), lab=list(lab), nu=nu, nt=nt, rank=rank,
                                      distinct=int(len(dist)), n_gaps=int(len(gr)),
                                      r_mean=float(gr.mean()) if len(gr) else float('nan'),
                                      smallest_kept=float(s[nz].min()) if rank else float('nan'),
                                      largest_dropped=float(s[~nz].max()) if rank < len(s) else 0.0, pooled=pooled))
            del Z
            log('    irrep %d %s: nu %d nt %d; basis %.0f s, Z_mu %.0f s, %d SVDs %.0f s; ranks %s'
                % (r, list(lab), nu, nt, t3 - t2, t4 - t3, len(D), tsv,
                   sorted(set(x['rank'] for d in D for x in d['rows'] if x['key'] == key))))

    # pooled per direction
    for d in D:
        g_pool = [f2.gap_ratios(values['%s__%s' % (d['name'], x['key'])]) for x in d['rows']
                  if x['pooled'] and x['distinct'] >= 20]
        g_pool = np.concatenate(g_pool) if g_pool else np.zeros(0)
        d['n_gaps'] = int(len(g_pool))
        d['r_mean'] = float(g_pool.mean()) if len(g_pool) else float('nan')
        d['r_sem'] = float(g_pool.std() / np.sqrt(len(g_pool))) if len(g_pool) else float('nan')
        d['rank_pooled'] = int(sum(x['rank'] for x in d['rows'] if x['pooled']))
        log('  %-9s delta %5.2f deg (%-8s %-4s)  <r> = %.4f +- %.4f on %d  ranks %s'
            % (d['name'], d['delta'], d['nearest'], d['nearest_kind'], d['r_mean'], d['r_sem'], d['n_gaps'],
               [x['rank'] for x in d['rows']]))

    res = dict(th=th, method='lean first-order probe, direction-independent (r, label) bases, M_c linear in q',
               pooling=('all characters' if th % 2 else 'non-trivial characters (trivial one carries the hidden Z_2)'),
               H_order=nH, irreps=irr, extra=names, dt=dt, du=du, subblocks=subs, walls=[(w[0], w[2]) for w in WALLS],
               directions=D, seconds=time.time() - T0)
    base = 'Wall_scan_E%d%s' % (th, tag)
    with open(os.path.join(OUT, base + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, base + '_values.npz'), **values)
    with open(os.path.join(OUT, 'Wall_scan.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.1f s]' % (time.time() - T0))
    return res


def parse_seeds(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    th = int(args[0])
    dirs = ['qrand%d' % s for s in parse_seeds(opts.get('seeds', '101-148'))]
    dirs += [x for x in opts.get('extra', 'qgen,qgen2').split(',') if x]
    qdirs = [(d['name'], d['q']) for d in json.load(open(opts['qfile']))['directions']] if 'qfile' in opts else []
    run(th, dirs, chunk=int(opts.get('chunk', 400)), with_trivial='with-trivial' in opts,
        direct_stab='direct-stab' in opts, qdirs=qdirs, tag=opts.get('tag', ''))
