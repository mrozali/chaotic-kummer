"""The gap-ratio deficit at E = 4 is a property of the level, not of the reference direction.

Backs the sentences on E = 4 at the end of "Statistical analysis" in the End Matter of the Letter:

  (a) over 3000 random directions more than 4 deg from every mirror the pooled <r> is 0.495 on average (the script
      also writes the spread between directions, 0.019, which the Letter does not quote; the reference direction
      has 0.471);
  (b) in the three sub-blocks without a hidden Z_2 the ten gap ratios at the TOP of each sub-block's spectrum
      average 0.40 and the remaining ones 0.53 (NOT the ten largest ratios);
  (c) the structure at the top.  What this script establishes: (c1) along the singlet each of these sub-blocks has a
      nondegenerate top level 6.1673 and an eightfold degenerate one, 5.6077 = pi sqrt((7 + sqrt 33)/4); (c2) at first
      order in the distance from the singlet the eightfold level splits in the pattern (sqrt2, 1, 1, 0, 0, -1, -1,
      -sqrt2), i.e. two single levels and three pairs; (c3) over directions far from the mirrors the highest level is
      isolated and the two below it are nearly degenerate (position-resolved gaps and gap ratios are written to the
      output).  It does NOT establish that the nine highest levels of a generic direction are the continuation of the
      singlet's nine: along the great circle to the reference direction they go through narrow avoided crossings
      with the levels below (the smallest gap between the 9th and the 10th level is reported);
  (d) no comparable structure at E = 3, 5 (and 6 with --levels=3,5,6, which takes several minutes);
  (e) no further exact symmetry in these sub-blocks: the first-order matrices at two random directions have a single
      joint invariant subspace (the test that finds the hidden Z_2 of the trivial character);
  (f) the two eigenspaces of the hidden Z_2 (50 and 26 values) give <r> = 0.45 each.

Method.  The first-order matrix is linear in the unit quaternion, M_c = sum_mu q_mu M_mu.  For every sub-block of the
generic stabiliser (character of H_c, F_R, Z_R) the four matrices Z_mu = M_mu Q are built once (Q an orthonormal basis
of the untwisted side of the sub-block, the twisted side compressed to the joint range of the four), as in
r2g_wall_scan.run; a direction then costs one small SVD per sub-block.  The hidden Z_2 of the trivial character is
found as in r2g_hidden_firstorder.py, from the eigenvectors of N(q) = (M_c Q)^dag (M_c Q) at two random directions.
Gate: the reference direction must reproduce the Letter's sub-block values and the pooled 0.471 on 294 ratios.

Usage:  python r2g_E4_deficit.py [--n=1000] [--seeds=20261116,7,12345] [--levels=3,5] [--nother=300]
Output: r2_out/E4_deficit.json and a log on stdout.  About one minute with the defaults.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_L6 as pl6
import r2g_generic_probe_mf as mf
import r2g_generic_probe_mf_lean as lean
import r2g_wall_scan as ws

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
E4I = np.eye(4)
WALLS = np.array([E4I[i] for i in range(4)]
                 + [(E4I[i] - s * E4I[j]) / np.sqrt(2) for i, j in itertools.combinations(range(4), 2) for s in (1, -1)])
Q_SINGLET = np.array([0.0, 0.0, -1.0, 0.0])


def wall_angle(q):
    """Angle (deg) to the nearest of the 16 mirrors q_mu = 0, q_mu = +-q_nu."""
    q = np.asarray(q, float)
    return float(np.degrees(np.arcsin(np.minimum(1.0, np.abs(WALLS @ (q / np.linalg.norm(q)))))).min())


def nz(s, tol=1e-7):
    s = np.sort(s)
    return s[s > tol * s[-1]]


def ratios(v):
    d = np.diff(np.sort(v))
    return np.minimum(d[1:], d[:-1]) / np.maximum(d[1:], d[:-1])


def svals(Z, q, U=None):
    A = np.tensordot(np.asarray(q, float) / np.linalg.norm(q), Z, axes=(0, 0))
    return nz(np.linalg.svd(A if U is None else A @ U, compute_uv=False))


def far_directions(rng, n, dmin=4.0):
    out = []
    while len(out) < n:
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        if wall_angle(q) > dmin:
            out.append(q)
    return out


# ---------------------------------------------------------------------------------------------------------------
def sub_blocks(th, chunk=400):
    """{key: Z} with Z[mu] = W^dag M_mu Q for every sub-block of the generic stabiliser at level th (W an orthonormal
    basis of the joint range of the four M_mu Q on the twisted side).  Follows r2g_wall_scan.run (same seeds)."""
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    c_ref = gp.cmat('qgen')
    Hc, sgn, rule = gp.stabiliser(S, c_ref, 3)
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    nH = len(Hc)
    Mref = mf.MFree(S, th, c_ref, chunk=chunk)
    dt, du = Mref.shape
    nW = S.nW[th]
    Mmu = [mf.MFree(S, th, ws.QB[m], chunk=chunk) for m in range(4)]
    act = lean.LeanAction(S, th)
    Ut = {gi: S.Utw[th][gi] for gi in Hc}

    rng = np.random.default_rng(11)
    syms = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xt_mat = S.Utw[th][gi]
        x = rng.standard_normal((du, 3)) + 1j * rng.standard_normal((du, 3))
        rx = act.apply(gi, x, right_only=True)
        lhs = pl6.apply_tw_right(Xt_mat, Mref @ x, nW)
        rhs = Mref @ rx
        nrm = max(float(np.max(np.abs(rhs))), 1e-30)
        cp = float(np.max(np.abs(lhs - rhs))) / nrm
        cm = float(np.max(np.abs(lhs + rhs))) / nrm
        s_ = +1 if cp < 1e-9 else (-1 if cm < 1e-9 else 0)
        if s_:
            syms[nm] = (gi, Xt_mat, s_)
    names = sorted(syms)

    def weights(r):
        return [dims[r] / nH * np.conj(chars[r, elem_class[k]]) for k in range(nH)]

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
                    hs = [syms[names[i]][0] for i in sub]
                    tu += w[k] * cu * act.trace_with_right(gi, hs)
                    B = Ut[gi]
                    for i in sub:
                        B = B @ syms[names[i]][1]
                    tt += w[k] * sgn[gi] * ct * trUt * np.trace(B)
        return int(np.rint(tu.real)), int(np.rint(tt.real))

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

    out = {}
    for r in irr:
        for lab in itertools.product((+1, -1), repeat=len(names)):
            nu, nt = dims_lab(r, lab)
            if nu == 0 or nt == 0:
                continue
            key = lean.labkey(r, lab)
            rs = np.random.default_rng(1000 * r + 17 * (1 + sum((1 + l) // 2 * 2 ** i for i, l in enumerate(lab))))
            Q = np.empty((du, nu), complex)
            for k0 in range(0, nu, chunk):
                k1 = min(nu, k0 + chunk)
                X = rs.standard_normal((du, k1 - k0)) + 1j * rs.standard_normal((du, k1 - k0))
                Q[:, k0:k1] = proj_lab(r, lab, X)
            err = lean.cholqr_inplace(Q)
            assert err < 1e-11, err
            Z = [Mmu[m].matvec(np.ascontiguousarray(Q)) for m in range(4)]
            U, sv, _ = np.linalg.svd(np.concatenate(Z, axis=1), full_matrices=False)
            W = U[:, :int((sv > 1e-9 * sv[0]).sum())]
            out[key] = np.array([W.conj().T @ z for z in Z])
    return out


def joint_subspaces(Z, qa, qb, thr=1e-8):
    """Orthonormal bases of the joint invariant subspaces (untwisted side) of N(qa) and N(qb), largest first."""
    A = np.tensordot(qa / np.linalg.norm(qa), Z, axes=(0, 0))
    B = np.tensordot(qb / np.linalg.norm(qb), Z, axes=(0, 0))
    ea, va = np.linalg.eigh(A.conj().T @ A)
    eb, vb = np.linalg.eigh(B.conj().T @ B)
    n = len(ea)
    adj = np.zeros((2 * n, 2 * n))
    adj[:n, n:] = np.abs(va.conj().T @ vb) > thr
    nc, lab = connected_components(adj + adj.T, directed=False)
    return [va[:, lab[:n] == c] for c in np.argsort(-np.bincount(lab[:n]))]


def top_and_rest(blocks, dirs, ntop=10):
    """Mean of the ntop gap ratios at the top of each sub-block's spectrum, and of the others, over directions."""
    top, rest = [], []
    for q in dirs:
        for Z in blocks:
            r = ratios(svals(Z, q))
            top.append(r[-ntop:])
            rest.append(r[:-ntop])
    return float(np.mean(np.concatenate(top))), float(np.mean(np.concatenate(rest)))


def main(n=1000, seeds=(20261116, 7, 12345), levels=(3, 5), nother=300):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q_ref = ws.quat(gp.cmat('qgen'))
    res = dict(q_ref=q_ref.tolist(), q_ref_wall_angle=wall_angle(q_ref))

    blk = sub_blocks(4)
    keys = list(blk)
    assert keys[0] == 'rho0_labpp' and len(keys) == 4, keys
    Z0, clean = blk[keys[0]], keys[1:]
    print('E = 4: sub-blocks %s  [%.0f s]' % ({k: blk[k].shape[1:] for k in keys}, time.time() - t0), flush=True)

    # ---- hidden Z_2 of the trivial character, and (e): no further joint invariant subspace anywhere
    rng = np.random.default_rng(seeds[0])
    bases = joint_subspaces(Z0, rng.standard_normal(4), rng.standard_normal(4))
    assert [b.shape[1] for b in bases] == [50, 26, 1], [b.shape[1] for b in bases]
    U50, U26 = bases[0], bases[1]
    comps = {k: [b.shape[1] for b in joint_subspaces(blk[k], rng.standard_normal(4), rng.standard_normal(4))] for k in clean}
    for U in (U50, U26):
        comps.setdefault('leaves', []).append([b.shape[1] for b in joint_subspaces(
            np.array([z @ U for z in Z0]), rng.standard_normal(4), rng.standard_normal(4))])
    assert all(len(comps[k]) == 1 for k in clean) and all(len(c) == 1 for c in comps['leaves']), comps
    res['joint_invariant_subspaces'] = dict(trivial_character=[50, 26, 1], others=comps)
    print('joint invariant subspaces at two random directions: trivial character 50 + 26 + 1 (common null vector), '
          'each eigenspace and each other sub-block a single component: %s' % comps, flush=True)

    def per_block(q):
        out = {k: ratios(svals(blk[k], q)) for k in clean}
        out['leaf50'] = ratios(svals(Z0, q, U50))
        out['leaf26'] = ratios(svals(Z0, q, U26))
        return out

    # ---- gate: the reference direction as in the Letter
    ref = per_block(q_ref)
    got = {k: round(float(v.mean()), 4) for k, v in ref.items()}
    want = dict(rho2_labpp=0.4248, rho3_labpp=0.5277, rho6_labpp=0.4898, leaf50=0.4541, leaf26=0.4149)
    pooled_ref = float(np.concatenate(list(ref.values())).mean())
    assert got == want, got
    assert abs(pooled_ref - 0.4711) < 1e-4 and sum(len(v) for v in ref.values()) == 294
    res['reference'] = dict(sub_blocks=got, pooled=pooled_ref, n_ratios=294)
    print('gate: reference direction %s, pooled %.4f on 294 ratios' % (got, pooled_ref), flush=True)

    # ---- (a), (b), (f): samples of far directions
    res['samples'] = []
    for seed in seeds:
        dirs = far_directions(np.random.default_rng(seed), n)
        pooled, top, rest, l50, l26 = [], [], [], [], []
        for q in dirs:
            b = per_block(q)
            pooled.append(np.concatenate(list(b.values())).mean())
            for k in clean:
                assert len(b[k]) == 74
                top.append(b[k][-10:])
                rest.append(b[k][:-10])
            l50.append(b['leaf50'].mean())
            l26.append(b['leaf26'].mean())
        pooled = np.array(pooled)
        row = dict(seed=seed, n_directions=n, min_angle_deg=4.0, pooled_mean=float(pooled.mean()),
                   pooled_sd_between_directions=float(pooled.std(ddof=1)),
                   fraction_below_reference=float((pooled <= pooled_ref).mean()),
                   clean_top10=float(np.mean(top)), clean_rest=float(np.mean(rest)),
                   leaf50=float(np.mean(l50)), leaf26=float(np.mean(l26)))
        res['samples'].append(row)
        print('seed %-9d %d directions > 4 deg: pooled %.4f, sd between directions %.4f, %.1f%% below the reference | '
              'three sub-blocks: ten ratios at the top %.3f, the others %.3f | eigenspaces %.3f (50), %.3f (26)'
              % (seed, n, row['pooled_mean'], row['pooled_sd_between_directions'], 100 * row['fraction_below_reference'],
                 row['clean_top10'], row['clean_rest'], row['leaf50'], row['leaf26']), flush=True)
    tot = {k: float(np.mean([r[k] for r in res['samples']])) for k in ('pooled_mean', 'pooled_sd_between_directions',
                                                                        'clean_top10', 'clean_rest', 'leaf50', 'leaf26')}
    res['all_samples'] = tot
    print('all %d directions: pooled %.3f, spread %.3f, top ten %.2f, others %.2f, eigenspaces %.2f and %.2f'
          % (n * len(seeds), tot['pooled_mean'], tot['pooled_sd_between_directions'], tot['clean_top10'], tot['clean_rest'],
             tot['leaf50'], tot['leaf26']), flush=True)

    # ---- (c): the top of the spectrum
    # (c1), (c2): the singlet and its neighbourhood, along the great circle from the singlet to the reference direction
    om = np.arccos(np.clip(Q_SINGLET @ q_ref, -1, 1))
    ts = np.linspace(0, 1, 4001)
    path = [(np.sin((1 - t) * om) * Q_SINGLET + np.sin(t * om) * q_ref) / np.sin(om) for t in ts]
    s8 = np.pi * np.sqrt((7 + np.sqrt(33)) / 4)
    pattern = np.array([np.sqrt(2), 1, 1, 0, 0, -1, -1, -np.sqrt(2)])
    i05 = int(np.searchsorted(ts * np.degrees(om), 0.5))
    x05 = ts[i05] * om
    res['singlet_top'] = {}
    for k in clean:
        s0 = svals(blk[k], Q_SINGLET)[::-1]
        assert abs(s0[0] - 6.1673) < 1e-3 and np.allclose(s0[1:9], s8, atol=1e-6) and s0[9] < 5.4, s0[:10]
        S = np.array([np.sort(svals(blk[k], q))[::-1][:12] for q in path[1:]])
        slopes = (S[i05 - 1][1:9] - s8) / x05
        c = slopes[0] / np.sqrt(2)
        assert np.abs(slopes / c - pattern).max() < 0.02, slopes / c
        gap = S[:, 8] - S[:, 9]                                     # between the 9th and the 10th level from the top
        res['singlet_top'][k] = dict(singlet_top10=s0[:10].tolist(), octet_slopes_over_c=(slopes / c).tolist(), c=float(c),
                                     min_gap_9_10_along_path=float(gap.min()),
                                     where_deg_from_singlet=float(np.degrees(ts[1:][gap.argmin()] * om)),
                                     reference_top12=S[-1].tolist())
        print('%s: singlet %.4f (x1), %.4f (x8), next %.4f | first-order splitting of the eightfold level, in units of '
              '%.3f |x|: %s | smallest gap between the 9th and 10th level on the way to the reference: %.4f at %.1f deg'
              % (k, s0[0], s0[1], s0[9], c, np.round(slopes / c, 3), gap.min(), res['singlet_top'][k]['where_deg_from_singlet']),
              flush=True)
        print('    reference direction, twelve highest levels: %s' % np.round(S[-1], 3), flush=True)
    # (c3): position-resolved gaps and gap ratios at the top, over directions far from the mirrors
    dirs = far_directions(np.random.default_rng(seeds[0] + 1), n)
    L = np.array([[np.sort(svals(blk[k], q))[::-1] for k in clean] for q in dirs])        # (direction, sub-block, 76)
    gaps = L[:, :, :-1] - L[:, :, 1:]
    rr = np.minimum(gaps[:, :, 1:], gaps[:, :, :-1]) / np.maximum(gaps[:, :, 1:], gaps[:, :, :-1])
    res['top_positions'] = dict(n_directions=n, mean_gap=gaps.mean(axis=(0, 1))[:12].tolist(),
                                r_by_position=rr.mean(axis=(0, 1))[:12].tolist(),
                                bulk_mean_gap=float(gaps[:, :, 20:60].mean()), bulk_r=float(rr[:, :, 20:60].mean()))
    print('%d directions > 4 deg, three sub-blocks, positions counted from the top:' % n)
    print('    mean gap between levels k and k+1, k = 1..10: %s  (bulk %.3f)'
          % (np.round(gaps.mean(axis=(0, 1))[:10], 3), res['top_positions']['bulk_mean_gap']))
    print('    mean gap ratio at level k+1,       k = 1..10: %s  (bulk %.3f)'
          % (np.round(rr.mean(axis=(0, 1))[:10], 3), res['top_positions']['bulk_r']), flush=True)

    # ---- (d): the same statistic at other levels (all sub-blocks without a hidden Z_2)
    res['other_levels'] = {}
    for th in levels:
        t1 = time.time()
        b = sub_blocks(th)
        use = [k for k in b if not (th % 2 == 0 and k.startswith('rho0'))]
        dirs = far_directions(np.random.default_rng(seeds[0] + th), nother)
        top, rest = top_and_rest([b[k] for k in use], dirs)
        res['other_levels'][str(th)] = dict(sub_blocks={k: list(b[k].shape[1:]) for k in use}, n_directions=nother,
                                            top10=top, rest=rest)
        print('E = %d: %d sub-blocks, %d directions > 4 deg: ten ratios at the top %.3f, the others %.3f  [%.0f s]'
              % (th, len(use), nother, top, rest, time.time() - t1), flush=True)

    res['seconds'] = time.time() - t0
    with open(os.path.join(OUT, 'E4_deficit.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote r2_out/E4_deficit.json  [%.0f s]' % res['seconds'])


if __name__ == '__main__':
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    main(n=int(opts.get('n', 1000)), seeds=tuple(int(x) for x in opts.get('seeds', '20261116,7,12345').split(',')),
         levels=tuple(int(x) for x in opts.get('levels', '3,5').split(',') if x), nother=int(opts.get('nother', 300)))
