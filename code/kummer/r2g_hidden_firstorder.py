"""r2g_hidden_firstorder.py -- the hidden Z_2 of the even levels from first-order data alone.

The involution that splits the trivial-character, Z_R = + sub-block of an even level does not depend on the
direction: its eigenspaces are common invariant subspaces of N(q) = (M_q Q)^+ (M_q Q) for every unit quaternion
q, where Q is the direction-independent basis of the untwisted (irrep 0, label (+,+)) space and M_q = sum_mu
q_mu M_mu (r2g_wall_scan.py).  So they follow from the first-order matrices of ONE level along TWO directions,
with no truncated Hamiltonian and no off-level matrix elements: the eigenvectors of N(q_a) and N(q_b) overlap
only inside a common invariant subspace, and the connected components of the overlap graph are the eigenspaces
(the Letter's completeness test with two directions in place of two couplings).

Method.  Q exactly as in the lean probe / wall scan (same projector, CholQR3), Z_mu = M_mu Q for the four
quaternion units, then the ten Hermitian matrices S_mu,nu = (Z_mu^+ Z_nu + Z_nu^+ Z_mu)/2, so that
N(q) = sum_mu,nu q_mu q_nu S_mu,nu for every direction at the cost of an nu x nu sum.  Singular values are
sqrt(eig N); on a leaf U they are sqrt(eig U^+ N(q) U).

Gates (the run exits non-zero if one fails):
  G1  the leaves found at thresholds 1e-7, 1e-6, 1e-5 coincide, and their sizes add up to the rank
  G2  every S_mu,nu is block diagonal on the leaves (the split is the same for every direction)
  G3  the common null space of the four units has dimension nu - rank (the zero modes do not depend on q)
  G4  for every direction with a stored record, the unresolved singular values reproduce the stored
      distinct values (Generic_probe_<dir>_E<th>_{mf,lean}_values.npz, key rho0_labpp)

Output: r2_out/Hidden_firstorder_E<th>.json, r2_out/Hidden_firstorder_E<th>_values.npz with the distinct values
(rounded to 1e-9, ascending) under '<dir>__leaf<k>' and '<dir>__unresolved'; with --save-gram also
r2_out/Hidden_firstorder_E<th>_gram.npy (10 x nu x nu, the pairs mu <= nu in row order) and
r2_out/Hidden_firstorder_E<th>_leaves.npz (the leaf index sets and the eigenvectors of N(q_a)).

Usage: python r2g_hidden_firstorder.py <th> [--dirs=qgen,qgen2,qrand101-120] [--chunk=400] [--save-gram]
Memory at E = 8: Q 23 GB + four Z_mu 47 GB, then 14 GB of Gram matrices (about 80 GB peak).
"""
import itertools
import json
import os
import sys
import time

import numpy as np

import f2_common as f2
import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_L6 as pl6
import r2g_generic_probe_mf as mf
import r2g_generic_probe_mf_lean as lean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE = 0.5307

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
QB = [np.eye(2, dtype=complex), 1j * SX, 1j * SY, 1j * SZ]       # c = sum_mu q_mu QB_mu, as in gp.cmat
PAIRS = [(m, n) for m in range(4) for n in range(m, 4)]

E4 = np.eye(4)
WALLS = [E4[i] for i in range(4)]
WALLS += [(E4[i] - s * E4[j]) / np.sqrt(2) for i, j in itertools.combinations(range(4), 2) for s in (1, -1)]
WALLS = np.array(WALLS)


def quat(c):
    q = np.array([np.trace(c).real / 2, ((c[0, 1] + c[1, 0]) / 2j).real, ((c[0, 1] - c[1, 0]) / 2).real,
                  ((c[0, 0] - c[1, 1]) / 2j).real])
    return q / np.linalg.norm(q)


def wall_angle(q):
    return float(np.degrees(np.arcsin(np.minimum(1.0, np.abs(WALLS @ q)))).min())


def components(adj):
    """Connected components of a bipartite graph given by a dense boolean na x nb matrix (labels of the rows)."""
    na, nb = adj.shape
    lab = -np.ones(na, int)
    k = 0
    while (lab < 0).any():
        rows = np.zeros(na, bool)
        rows[np.argmax(lab < 0)] = True
        while True:
            cols = adj[rows].any(axis=0)
            new = adj[:, cols].any(axis=1) | rows
            if new.sum() == rows.sum():
                break
            rows = new
        lab[rows] = k
        k += 1
    return lab


def stats(s):
    d = np.unique(np.round(s, 9))
    g = f2.gap_ratios(d) if len(d) >= 20 else np.zeros(0)
    return d, (float(g.mean()) if len(g) else float('nan')), (float(g.std() / np.sqrt(len(g))) if len(g) else float('nan')), int(len(g))


def run(th, dirs, chunk=400, save_gram=False):
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()

    def log(s):
        print('[%7.0f s] %s' % (time.time() - T0, s), flush=True)

    assert th % 2 == 0, 'the hidden Z_2 acts at the even levels'
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built' % th)
    c_ref = gp.cmat('qgen')
    Hc, sgn, rule = gp.stabiliser(S, c_ref, 3)
    assert rule['Yb c Yb^T'] == len(Hc), rule
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    nH = len(Hc)
    r0 = irr[0]
    assert r0 == 0 and dims[r0] == 1 and np.allclose(chars[r0], 1), 'irrep 0 is the trivial character'
    Mref = mf.MFree(S, th, c_ref, chunk=chunk)
    dt, du = Mref.shape
    nW = S.nW[th]
    Mmu = [mf.MFree(S, th, QB[m], chunk=chunk) for m in range(4)]
    act = lean.LeanAction(S, th)
    Ut = {gi: S.Utw[th][gi] for gi in Hc}
    log('M %d x %d (twisted x untwisted), stabiliser order %d' % (dt, du, nH))

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
                assert s_ != 0, nm
                syms[nm] = (gi, Xt_mat, s_)
            else:
                assert s_ == syms[nm][2], (nm, lbl, cp, cm)
    names = sorted(syms)
    assert names == ['F_R', 'Z_R']
    lab = (+1, +1)
    key = lean.labkey(r0, lab)
    assert key == 'rho0_labpp', key

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

    nu, nt = dims_lab(r0, lab)
    log('%s: nu %d, nt %d' % (key, nu, nt))
    rs = np.random.default_rng(1000 * r0 + 17 * (1 + sum((1 + l) // 2 * 2 ** i for i, l in enumerate(lab))))
    Q = np.empty((du, nu), complex)
    for k0 in range(0, nu, chunk):
        k1 = min(nu, k0 + chunk)
        X = rs.standard_normal((du, k1 - k0)) + 1j * rs.standard_normal((du, k1 - k0))
        Q[:, k0:k1] = proj_lab(r0, lab, X)
    err = lean.cholqr_inplace(Q)
    assert err < 1e-11, err
    chk = proj_lab(r0, lab, np.ascontiguousarray(Q[:, :4]))
    assert float(np.max(np.abs(chk - Q[:, :4]))) < 1e-8
    log('basis Q %d x %d, orthonormality %.1e' % (du, nu, err))

    Z = []
    for m in range(4):
        Zm = np.empty((dt, nu), complex)
        for k0 in range(0, nu, chunk):
            k1 = min(nu, k0 + chunk)
            Zm[:, k0:k1] = Mmu[m].matvec(np.ascontiguousarray(Q[:, k0:k1]))
        Z.append(Zm)
        log('Z_%d done' % m)
    del Q

    Sg = np.zeros((len(PAIRS), nu, nu), complex)
    rows = 8192
    for i0 in range(0, dt, rows):
        B = [z[i0:i0 + rows] for z in Z]
        Bh = [b.conj().T for b in B]
        for p, (m, n) in enumerate(PAIRS):
            Sg[p] += Bh[m] @ B[n]
    del Z, B, Bh
    for p in range(len(PAIRS)):
        Sg[p] = (Sg[p] + Sg[p].conj().T) / 2
    log('Gram matrices S_mu,nu (%d x %d x %d)' % Sg.shape)
    if save_gram:
        np.save(os.path.join(OUT, 'Hidden_firstorder_E%d_gram.npy' % th), Sg)
        log('Gram matrices saved')

    def N_of(q):
        N = np.zeros((nu, nu), complex)
        for p, (m, n) in enumerate(PAIRS):
            N += (q[m] * q[n] * (1.0 if m == n else 2.0)) * Sg[p]
        return N

    # ---- common null space of the four units (G3)
    eg = np.linalg.eigvalsh(sum(Sg[p] for p, (m, n) in enumerate(PAIRS) if m == n))
    n_null = int((eg < 1e-9 * eg[-1]).sum())
    log('common null vectors of the four quaternion units: %d' % n_null)

    # ---- the finder: two random directions away from the mirrors
    rq = np.random.default_rng(20261116)
    two = []
    while len(two) < 2:
        q = rq.standard_normal(4)
        q /= np.linalg.norm(q)
        if wall_angle(q) > 4.0:
            two.append(q)
    ea, va = np.linalg.eigh(N_of(two[0]))
    eb, vb = np.linalg.eigh(N_of(two[1]))
    ka, kb = ea > 1e-9 * ea[-1], eb > 1e-9 * eb[-1]
    rank = int(ka.sum())
    assert rank == int(kb.sum()), (rank, int(kb.sum()))
    g3 = bool(n_null == nu - rank)
    va, vb = va[:, ka], vb[:, kb]
    O = np.abs(va.conj().T @ vb)
    del vb
    hist = {('%.0e' % t): int((O < t).sum()) for t in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)}
    log('rank %d; overlaps below threshold: %s of %d' % (rank, hist, O.size))
    labs, sizes = {}, {}
    for t in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        labs[t] = components(O > t)
        sizes[t] = sorted(np.bincount(labs[t]).tolist(), reverse=True)
        log('  threshold %.0e: %d components, sizes %s' % (t, len(sizes[t]), sizes[t][:8]))
    labm = labs[1e-6]
    order = np.argsort(-np.bincount(labm))
    leaves = [np.where(labm == c)[0] for c in order]
    g1 = bool(sizes[1e-7] == sizes[1e-6] == sizes[1e-5] and sum(len(l) for l in leaves) == rank
              and all(set(np.where(labs[t] == labs[t][l[0]])[0]) == set(l) for t in (1e-7, 1e-5) for l in leaves))
    del O
    U = [np.ascontiguousarray(va[:, l]) for l in leaves]
    log('leaves: %s' % [u.shape[1] for u in U])

    # ---- G2: every S_mu,nu block diagonal on the leaves
    scale = float(max(np.abs(Sg[p]).max() for p in range(len(PAIRS))))
    cross = 0.0
    for p in range(len(PAIRS)):
        SU = [Sg[p] @ u for u in U]
        for i in range(len(U)):
            for j in range(len(U)):
                if i != j:
                    cross = max(cross, float(np.abs(U[i].conj().T @ SU[j]).max()))
        del SU
    g2 = bool(cross < 1e-6 * scale)
    log('G2: max cross element of the ten S_mu,nu between leaves %.1e (scale %.1e)' % (cross, scale))
    if save_gram:
        np.savez(os.path.join(OUT, 'Hidden_firstorder_E%d_leaves.npz' % th), va=va, two=np.array(two),
                 **{'leaf%d' % i: l for i, l in enumerate(leaves)})

    # ---- directions
    values, rows_out, g4, g4_checked = {}, [], True, 0
    for name in dirs:
        q = quat(gp.cmat(name))
        N = N_of(q)
        e = np.linalg.eigvalsh(N)
        s = np.sqrt(np.clip(e[-rank:], 0, None))          # the common null space (G3) dropped, nothing else
        d_un, r_un, sem_un, n_un = stats(s)
        row = dict(name=name, q=q.tolist(), wall_angle=wall_angle(q), rank=int(len(s)), unresolved=dict(
            distinct=int(len(d_un)), r_mean=r_un, r_sem=sem_un, n_gaps=n_un), leaves=[])
        values['%s__unresolved' % name] = d_un
        for fn in ('Generic_probe_%s_E%d_mf_values.npz' % (name, th), 'Generic_probe_%s_E%d_lean_values.npz' % (name, th)):
            path = os.path.join(OUT, fn)
            if os.path.exists(path):
                zz = np.load(path)
                if key in zz.files:
                    ref = np.unique(np.round(zz[key], 9))          # stored distinct values: each must be found
                    j = np.clip(np.searchsorted(s, ref), 1, len(s) - 1)
                    dev = float(np.max(np.minimum(np.abs(s[j] - ref), np.abs(s[j - 1] - ref)))) if len(ref) <= len(s) else float('inf')
                    row['ref_distinct'] = int(len(ref))
                    row['ref_file'], row['ref_dev'] = fn, dev
                    g4 = g4 and dev < 1e-6
                    g4_checked += 1
                    break
        pooled, raw = [], []
        for i, u in enumerate(U):
            el = np.linalg.eigvalsh(u.conj().T @ (N @ u))
            sl_ = np.sqrt(np.clip(el, 0, None))
            raw.append(sl_)
            d, r, sem, ng = stats(sl_)
            leak = float(np.linalg.norm(N @ u - u @ (u.conj().T @ (N @ u))) / np.linalg.norm(N @ u))
            row['leaves'].append(dict(size=int(u.shape[1]), distinct=int(len(d)), r_mean=r, r_sem=sem, n_gaps=ng,
                                      s_min=float(sl_.min()), leakage=leak))
            values['%s__leaf%d' % (name, i)] = d
            pooled.append(f2.gap_ratios(d))
        union = np.sort(np.concatenate(raw))
        row['union_dev'] = float(np.max(np.abs(union - s))) if len(union) == len(s) else float('inf')
        pp = np.concatenate(pooled)
        row['leaves_pooled'] = dict(r_mean=float(pp.mean()), r_sem=float(pp.std() / np.sqrt(len(pp))), n_gaps=int(len(pp)))
        rows_out.append(row)
        log('%-9s (%.2f deg) unresolved <r> %.4f | leaves %s | leaves pooled %.4f +- %.4f | ref dev %s, union dev %.1e, leakage %.1e'
            % (name, row['wall_angle'], r_un, ' '.join('%d: %.4f' % (x['size'], x['r_mean']) for x in row['leaves']),
               row['leaves_pooled']['r_mean'], row['leaves_pooled']['r_sem'],
               ('%.1e' % row['ref_dev']) if 'ref_dev' in row else 'n/a', row['union_dev'],
               max(x['leakage'] for x in row['leaves'])))

    gates = dict(G1_leaves_stable=g1, G2_block_diagonal=g2, G3_common_null_space=g3, G4_stored_values=bool(g4),
                 G4_directions_checked=g4_checked)
    res = dict(th=th, key=key, nu=nu, nt=nt, rank=rank, n_null=n_null, finder_directions=[q.tolist() for q in two],
               overlap_histogram=hist, component_sizes={('%.0e' % t): v for t, v in sizes.items()},
               leaf_sizes=[int(u.shape[1]) for u in U], cross_max=cross, cross_scale=scale, gates=gates,
               directions=rows_out, seconds=time.time() - T0)
    with open(os.path.join(OUT, 'Hidden_firstorder_E%d.json' % th), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, 'Hidden_firstorder_E%d_values.npz' % th), **values)
    log('gates: %s' % gates)
    ok = g1 and g2 and g3 and g4
    log('ALL GATES PASS' if ok else 'GATE FAILURE')
    return ok


def parse_dirs(s):
    out = []
    for part in s.split(','):
        if part.startswith('qrand') and '-' in part:
            a, b = part[5:].split('-')
            out += ['qrand%d' % k for k in range(int(a), int(b) + 1)]
        elif part:
            out.append(part)
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    ok = run(int(args[0]), parse_dirs(opts.get('dirs', 'qgen,qgen2,qrand101-120')), chunk=int(opts.get('chunk', 400)),
             save_gram='save-gram' in opts)
    sys.exit(0 if ok else 1)
