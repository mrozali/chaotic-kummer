"""r2g_hidden_lean.py -- the hidden Z_2 of the even levels for further random directions, without the dense
D_0 of r2g_generic_sweep.py (13540 x 28478 complex, 6.2 GB at th_max = 6).

Same object and same resolution as the truncated-Hamiltonian Fir pipeline (r2g_generic_sweep.py, then
r2g_first_order_from_sweep.py, both at th_max = 6), restricted to the one block that carries the hidden Z_2:
the trivial character of H_c with the right-only labels (F_R, Z_R) = (1, 1) (block 1, key '(0, (1, 1))',
n = 4515 states at th_max = 6).  The other blocks have no hidden Z_2 (truncated-Hamiltonian runs: one leaf each),
and at the
even levels their first-order sub-blocks are the three non-trivial characters of the wall scan.

  1. Space(6), the stabiliser H_c, its characters and the extra Z_2's (Z_R, F_R, Pi; commute or anticommute
     with D_0 level by level) exactly as in the sweep.
  2. The block's bases level by level, untwisted and twisted, by a randomised range finder on the composite
     projector P_0 (1 + f F_R)/2 (1 + z Z_R)/2 (twisted side with the sweep's sign convention: label f of an
     anticommuting Z_2 is the eigenvalue of -X_t), the number of columns fixed by the projector's trace.  The
     sweep took the range of P_0 and split it by eigh; the subspace is the same.
  3. The block's D_0 matrix over all level pairs, Qt^H D_0 Qu, with D_0 applied matrix-free (MPair: the
     sector-by-sector product of r2g_generic_probe_mf.MFree for a pair (th_t, th_u); checked against the
     sweep's dense r2g_generic_sweep.M_pair on every pair with th_t, th_u <= 5 and on (6, 5), (5, 6)).
  4. r2c_gate19_lib.fully_resolve on SubSector(E0, M, nu) -> leaves (the truncated-Hamiltonian code, unchanged).
  5. Per leaf and level: the level-diagonal singular values (threshold 1e-9 max|M|, distinct values rounded
     to 1e-9; r2g_first_order_from_sweep.level_table).
  6. Fully resolved <r>: at E = 4, 6 the leaves pooled with the three non-trivial characters (rho 2, 3, 6,
     label (1, 1)) from Wall_scan_E<th>_values.npz, gap ratios on sub-blocks with >= 20 distinct values;
     at E = 3, 5 the scan's pooling is already fully resolved (Wall_scan_E3/E5.json, checked against the
     leaves of the truncated-Hamiltonian runs to 2e-10).  The unsplit block pooled the same way
     gives the sweep's 'blocks' column.
Validation: for a direction with a file of the truncated-Hamiltonian runs
r2_out/scan_L6/Generic_firstorder_<dir>_L6.json the leaf
sizes, the per-level rank and distinct counts of both leaves and the pooled <r> (leaves and blocks, E = 3..6)
are compared with it; the run fails on a mismatch.

Output: r2_out/Hidden_lean_<dir>_L6.json, r2_out/Hidden_lean_<dir>_L6_values.npz (distinct values per leaf
        and level), log r2_out/hidden_lean_<dir>_L6.log; the block and its leaves (E0, M, nu) are cached in
        r2_out/Hidden_lean_<dir>_L6_leaves.npz (~0.2 GB, reused unless --force)
Usage:  python r2g_hidden_lean.py <dir>[,<dir>...] [--check-pairs] [--force]
Runs of 2026-11-05 (2 cores, single-threaded BLAS, two directions at a time with one
fully_resolve at a time): ~17 min per direction (Space 40 s, bases ~6 min, D_0 block 30 s, fully_resolve ~10 min),
peak memory ~3 GB per run.  qrand101 (with --check-pairs) reproduces the leaves of the truncated-Hamiltonian runs:
sizes [2885, 1630],
identical rank / distinct / nu / nt per level, pooled <r> (leaves and blocks, E = 3..6) to 3e-11.  qrand117-120: the
same split (components 2583 + 1328 at lambda = 0.2, 0.35; both leaves irreducible), including qrand117 at 0.001 deg
from the wall q0 = 0 (plateau 10 dex instead of 12-13).
"""
import fcntl
import json
import os
import sys
import time

import numpy as np

import f2_common as f2
import r2_d0 as d0
import r2c_gate19_lib as L
import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_mf as mf
import r2g_generic_sweep as gs
from r2_algebra import PLUS, MINUS
from r2c_build import SubSector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
TH_MAX = int(os.environ.get('HIDDEN_LEAN_TH', 6))   # 6; smaller values only to test the code paths
L.LAM_T1 = [0.2, 0.35]  # find_split diagonalises at every LAM_T1 value but uses only 0.2 and 0.35 (0.5 is discarded)
LABEL = (1, 1)          # (F_R, Z_R) of the block that carries the hidden Z_2
NONTRIV = (2, 3, 6)     # the non-trivial characters, label (1, 1), from the wall scan


class MPair(mf.MFree):
    """D_0 (twisted th_t <- untwisted th_u) along c, matrix-free; = r2g_generic_sweep.M_pair(S, th_t, th_u, c) @ X."""

    def __init__(self, S, th_t, th_u, c, chunk=200):
        self.A = {Ad: S.A[th_u].get((th_t, Ad)) for Ad in (PLUS, MINUS)}
        self.nW = S.nW.get(th_t, 0)
        self.empty = self.nW == 0 or self.A[PLUS] is None or self.A[MINUS] is None
        self.nV = 0 if self.empty else self.A[PLUS].shape[1]
        self.off, self.du = S.sect_offsets(th_u)
        self.dt = self.nW ** 2
        self.sect = list(S.sect[th_u])
        self.voff, self.vdim = S.voff[th_u], S.vdim[th_u]
        self.c = c
        self.chunk = chunk
        self.terms = [(Ad, Bd) for Ad in (PLUS, MINUS) for Bd in (PLUS, MINUS) if abs(c[Ad, Bd]) > 1e-14]
        self.Bds = sorted(set(Bd for Ad, Bd in self.terms))
        self.shape = (self.dt, self.du)

    def matvec(self, X):
        if self.empty:
            return np.zeros((self.dt, X.shape[1]), complex)
        return super().matvec(X)


def check_pairs(S, c, levels, log):
    rng = np.random.default_rng(5)
    worst = 0.0
    for th_u in levels:
        for th_t in levels:
            big = max(th_t, th_u) == 6
            if big and (th_t, th_u) not in ((6, 5), (5, 6)):
                continue
            X = rng.standard_normal((S.sect_offsets(th_u)[1], 3)) + 1j * rng.standard_normal((S.sect_offsets(th_u)[1], 3))
            ref = gs.M_pair(S, th_t, th_u, c) @ X
            got = MPair(S, th_t, th_u, c) @ X
            dev = float(np.max(np.abs(ref - got))) if ref.size else 0.0
            worst = max(worst, dev / max(float(np.max(np.abs(ref))) if ref.size else 1.0, 1e-30))
    log('MPair against the dense M_pair (all pairs <= 5, (6,5), (5,6)): max relative deviation %.1e' % worst)
    assert worst < 1e-12, worst


def build_block(S, c, log, check=False):
    g = S.g
    levels = [th for th in S.levels if S.nW.get(th, 0) > 0 and S.sect_offsets(th)[1] > 0]
    du = {th: S.sect_offsets(th)[1] for th in levels}
    if check:
        check_pairs(S, c, levels, log)
    Hc, sgn, rule = gp.stabiliser(S, c, min(3, TH_MAX))
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    nH = len(Hc)
    assert nH == 16, nH
    assert np.allclose(chars[0], 1.0) and dims[0] == 1 and phys[0], 'irrep 0 is not the trivial character'
    w = [dims[0] / nH * np.conj(chars[0, elem_class[k]]) for k in range(nH)]
    log('stabiliser order %d (%d with G), irreps %d, usable %s' % (
        nH, sum(1 for v in sgn.values() if v < 0), len(dims), [r for r in range(len(dims)) if phys[r]]))
    rng = np.random.default_rng(23)

    # extra Z_2's, classified on the level-diagonal blocks as in the sweep
    cand = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        cand[nm] = ({th: gs.right_only_sparse(S, th, gi) for th in levels},
                    {th: ('right', S.Utw[th][gi]) for th in levels})
    cand['Pi'] = ({th: gs.pi_untw_sparse(S, th) for th in levels}, {th: ('pi', None) for th in levels})
    syms = {}
    for nm, (Xu, Xt) in cand.items():
        cp = cm = 0.0
        for th in levels:
            Mth = MPair(S, th, th, c)
            x = rng.standard_normal((du[th], 3)) + 1j * rng.standard_normal((du[th], 3))
            kind, U = Xt[th]
            lhs = gs.tw_apply(kind, U, Mth @ x, S.nW[th])
            rhs = Mth @ (Xu[th] @ x)
            nrm = max(float(np.max(np.abs(rhs))), 1e-30)
            cp = max(cp, float(np.max(np.abs(lhs - rhs))) / nrm)
            cm = max(cm, float(np.max(np.abs(lhs + rhs))) / nrm)
        if cp < 1e-9:
            syms[nm] = +1
        elif cm < 1e-9:
            syms[nm] = -1
        log('    %s: commutes %.1e, anticommutes %.1e -> %s' % (
            nm, cp, cm, ('symmetry (x G)' if syms.get(nm) == -1 else 'symmetry') if nm in syms else 'not a symmetry'))
    names = sorted(syms)
    assert names == ['F_R', 'Z_R'] and syms['F_R'] == -1 and syms['Z_R'] == +1, syms
    f, z = LABEL

    Qu, Qt, Eu, Et = [], [], [], []
    for th in levels:
        nW = S.nW[th]
        Uu = {gi: gs.U_untw_sparse(S, th, gi) for gi in Hc}
        Fu, Zu = cand['F_R'][0][th], cand['Z_R'][0][th]
        VF, VZ = S.Utw[th][g.i_zF], S.Utw[th][g.i_z]
        sF, sZ = f * syms['F_R'], z * syms['Z_R']          # twisted label convention of split_level
        # traces of the composite projectors
        tu = tt = 0.0
        for k, gi in enumerate(Hc):
            U = Uu[gi]
            tu += w[k] * 0.25 * complex(U.diagonal().sum() + f * (U @ Fu).diagonal().sum()
                                        + z * (U @ Zu).diagonal().sum() + f * z * (U @ Fu @ Zu).diagonal().sum())
            Ut = S.Utw[th][gi]
            trU = np.trace(Ut)
            tt += w[k] * sgn[gi] * 0.25 * complex(trU * (np.trace(Ut) + sF * np.trace(Ut @ VF) + sZ * np.trace(Ut @ VZ)
                                                         + sF * sZ * np.trace(Ut @ VF @ VZ)))
        mu, mt = int(np.rint(tu.real)), int(np.rint(tt.real))
        assert abs(tu - mu) < 1e-6 and abs(tt - mt) < 1e-6, (th, tu, tt)

        def range_basis(m, n, apply, chunk=256):
            if m <= 0:
                return np.zeros((n, 0), complex)
            X = rng.standard_normal((n, m + 8)) + 1j * rng.standard_normal((n, m + 8))
            Y = np.empty_like(X)
            for i0 in range(0, X.shape[1], chunk):   # column by column, so the peak memory stays ~2 n (m + 8)
                Y[:, i0:i0 + chunk] = apply(X[:, i0:i0 + chunk])
            del X
            Q, R = np.linalg.qr(Y)
            del Y
            keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])
            assert int(keep.sum()) == m, (th, m, int(keep.sum()))
            return Q[:, keep]

        def apply_u(X):
            X = (X + z * (Zu @ X)) / 2
            X = (X + f * (Fu @ X)) / 2
            acc = np.zeros_like(X)
            for k, gi in enumerate(Hc):
                acc += w[k] * (Uu[gi] @ X)
            return acc

        def apply_t(X):
            X = (X + sZ * gs.tw_apply('right', VZ, X, nW)) / 2
            X = (X + sF * gs.tw_apply('right', VF, X, nW)) / 2
            acc = np.zeros_like(X)
            for k, gi in enumerate(Hc):
                acc += w[k] * sgn[gi] * gs.tw_apply('both', S.Utw[th][gi], X, nW)
            return acc

        qu = range_basis(mu, du[th], apply_u)
        qt = range_basis(mt, nW * nW, apply_t)
        # the basis must be invariant: P q = q
        if qu.shape[1]:
            assert np.max(np.abs(apply_u(qu[:, :4]) - qu[:, :4])) < 1e-10
        if qt.shape[1]:
            assert np.max(np.abs(apply_t(qt[:, :4]) - qt[:, :4])) < 1e-10
        Qu.append((th, qu))
        Qt.append((th, qt))
        Eu.append(np.full(mu, float(th)))
        Et.append(np.full(mt, float(th)))
        log('  level %d: untwisted %d of %d, twisted %d of %d' % (th, mu, du[th], mt, nW * nW))
    Eu, Et = np.concatenate(Eu), np.concatenate(Et)
    nu, nt = len(Eu), len(Et)
    cu = np.concatenate([[0], np.cumsum([q.shape[1] for th, q in Qu])])
    ct = np.concatenate([[0], np.cumsum([q.shape[1] for th, q in Qt])])
    Ms = np.zeros((nt, nu), complex)
    t0 = time.time()
    for iu, (th_u, qu) in enumerate(Qu):
        if qu.shape[1] == 0:
            continue
        for it, (th_t, qt) in enumerate(Qt):
            if qt.shape[1] == 0:
                continue
            P = MPair(S, th_t, th_u, c)
            if P.empty:
                continue
            Ms[ct[it]:ct[it + 1], cu[iu]:cu[iu + 1]] = qt.conj().T @ (P @ qu)
    log('block %s: n = %d (untwisted %d, twisted %d); D_0 block over all level pairs in %.0f s, max|M| %.4f'
        % (str((0, LABEL)), nu + nt, nu, nt, time.time() - t0, float(np.max(np.abs(Ms)))))
    return SubSector(np.concatenate([Eu, Et]), Ms, nu)


def level_rows(sub, gmax):
    rows, dist = {}, {}
    for th in range(1, TH_MAX + 1):
        iu = sub.E0[:sub.nu] == th
        it = sub.E0[sub.nu:] == th
        nu_th, nt_th = int(iu.sum()), int(it.sum())
        if nu_th == 0 or nt_th == 0:
            rows[th] = dict(nu=nu_th, nt=nt_th, rank=0, distinct=0)
            continue
        sv = np.linalg.svd(sub.M[it][:, iu], compute_uv=False)
        kept = sv[sv > 1e-9 * gmax]
        drop = sv[sv <= 1e-9 * gmax]
        d = np.unique(np.round(kept, 9))
        rows[th] = dict(nu=nu_th, nt=nt_th, rank=int(len(kept)), distinct=int(len(d)),
                        smallest_kept=float(kept.min()) if len(kept) else None,
                        largest_dropped=float(drop.max()) if len(drop) else 0.0)
        dist[th] = d
    return rows, dist


def pooled(value_lists):
    gaps = [f2.gap_ratios(v) for v in value_lists if len(v) >= 20]
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    return float(np.mean(gaps)), float(np.std(gaps) / np.sqrt(len(gaps))), int(len(gaps))


def scan_entry(th, name):
    d = json.load(open(os.path.join(OUT, 'Wall_scan_E%d.json' % th)))
    e = [x for x in d['directions'] if x['name'] == name]
    return e[0] if e else None


def run(direction, check=False, force=False):
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    T0 = time.time()
    c = gp.cmat(direction)
    log('# Hidden Z_2, lean (trivial-character block only), %s, th_max %d  [%s]' % (
        direction, TH_MAX, time.strftime('%Y-%m-%d %H:%M')))
    lp = os.path.join(OUT, 'Hidden_lean_%s_L%d_leaves.npz' % (direction, TH_MAX))
    if os.path.exists(lp) and not force:
        d = np.load(lp, allow_pickle=False)
        sub = SubSector(d['E0_block'], d['M_block'], int(d['nu_block']))
        leaves = [(str(d['tag_%d' % k]), SubSector(d['E0_%d' % k], d['M_%d' % k], int(d['nu_%d' % k])), None)
                  for k in range(int(d['n_leaves']))]
        infos = json.loads(str(d['infos']))
        log('loaded the block and %d leaves from %s' % (len(leaves), os.path.basename(lp)))
    else:
        S = sp.Space(TH_MAX, mode='prim', verbose=False)
        log('Space built in %.0f s' % (time.time() - T0))
        sub = build_block(S, c, log, check=check)
        del S
        with open(os.path.join(OUT, '.hidden_lean_fully_resolve.lock'), 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)           # concurrent runs: one fully_resolve (~3 GB) at a time
            t1 = time.time()
            lg = []
            leaves = L.fully_resolve(sub, TH_MAX, tag='b1', log=lg)
        log('fully_resolve: n=%d -> %d leaves of sizes %s  [%.0f s]' % (sub.n, len(leaves), [x[1].n for x in leaves],
                                                                       time.time() - t1))
        infos = []
        for tag, info, n in lg:
            if isinstance(info, dict):
                infos.append(dict(tag=tag, **{k: (v.item() if isinstance(v, (np.generic,)) else v)
                                              for k, v in info.items()}))
            else:
                infos.append(dict(tag=tag, n=n, note=str(info)))
        npz = dict(E0_block=sub.E0, M_block=sub.M, nu_block=np.array(sub.nu), n_leaves=np.array(len(leaves)),
                   infos=np.array(json.dumps(infos)))
        for k, (tag, s, depth) in enumerate(leaves):
            npz['tag_%d' % k] = np.array(tag)
            npz['E0_%d' % k] = s.E0
            npz['M_%d' % k] = s.M
            npz['nu_%d' % k] = np.array(s.nu)
        np.savez(lp, **npz)
    for info in infos:
        if 'n_components' in info:
            log('  %s (n=%d): components %d sizes %s, singles %d, unlifted %d, thresh 1e%.0f, plateau %.1f dex, stable %s'
                % (info['tag'], info['n'], info['n_components'], info['comp_sizes'][:6], info['n_singles'],
                   info['n_unlifted'], np.log10(max(info['thresh_used'], 1e-300)), info['gap_dex'], info['stable']))
        else:
            log('  %s: %s (n=%d)' % (info['tag'], info['note'], info['n']))
    gmax = float(np.max(np.abs(sub.M)))
    block_rows, block_dist = level_rows(sub, gmax)
    leaf_rows, leaf_dist = {}, {}
    for tag, s, depth in leaves:
        leaf_rows[tag], leaf_dist[tag] = level_rows(s, gmax)
    res = dict(direction=direction, q=None, th_max=TH_MAX, block='(0, (1, 1))', n_block=sub.n,
               leaf_tags=[t for t, s, dpt in leaves], leaf_sizes=[s.n for t, s, dpt in leaves], find_split=infos,
               lam_t1=list(L.LAM_T1),
               block_rows={str(k): v for k, v in block_rows.items()},
               leaf_rows={t: {str(k): v for k, v in r.items()} for t, r in leaf_rows.items()})
    # pooled <r>
    r_full, r_blocks = {}, {}
    for th in (3, 4, 5, 6):
        e = scan_entry(th, direction)
        if e is None or th > TH_MAX:
            continue
        res['q'] = e['q']
        res['delta'] = e['delta']
        res['nearest'] = e['nearest']
        if th % 2:
            r_full[th] = dict(r_mean=e['r_mean'], r_sem=e['r_sem'], n_gaps=e['n_gaps'], source='Wall_scan_E%d.json' % th)
            r_blocks[th] = r_full[th]
            continue
        z = np.load(os.path.join(OUT, 'Wall_scan_E%d_values.npz' % th))
        other = [np.sort(z['%s__rho%d_labpp' % (direction, r)]) for r in NONTRIV]
        assert [len(v) for v in other] == [x['distinct'] for x in e['rows']], 'scan values vs json'
        lv = [leaf_dist[t][th] for t in res['leaf_tags'] if th in leaf_dist[t]]
        m, s, n = pooled(lv + other)
        r_full[th] = dict(r_mean=m, r_sem=s, n_gaps=n, source='leaves + Wall_scan_E%d_values.npz' % th)
        m, s, n = pooled([block_dist[th]] + other)
        r_blocks[th] = dict(r_mean=m, r_sem=s, n_gaps=n)
        log('  E=%d: scan (3 characters) %.4f | + block unsplit %.4f | fully resolved %.4f +- %.4f (%d gaps)'
            % (th, e['r_mean'], r_blocks[th]['r_mean'], r_full[th]['r_mean'], r_full[th]['r_sem'], n))
    res['r_full'] = {str(k): v for k, v in r_full.items()}
    res['r_blocks'] = {str(k): v for k, v in r_blocks.items()}
    # validation against the leaves of the truncated-Hamiltonian runs, where they exist
    ref_path = os.path.join(OUT, 'scan_L6', 'Generic_firstorder_%s_L6.json' % direction)
    if os.path.exists(ref_path) and TH_MAX == 6:
        ref = json.load(open(ref_path))
        ok = True
        ref_tags = [t for t in ref['leaf_tags'] if t.startswith('(0, (1, 1))')]
        ref_sizes = [ref['leaf_sizes'][ref['leaf_tags'].index(t)] for t in ref_tags]
        ok &= ref_sizes == res['leaf_sizes']
        log('  validation: leaf sizes %s vs September %s' % (res['leaf_sizes'], ref_sizes))
        dev = 0.0
        for th in (3, 4, 5, 6):
            for col, mine in (('leaves', r_full), ('blocks', r_blocks)):
                if th in mine:
                    dev = max(dev, abs(mine[th]['r_mean'] - ref[col][str(th)]['r_mean']))
            for t_ref, t_mine in zip(ref_tags, res['leaf_tags']):
                rr = [x for x in ref['leaves'][str(th)]['rows'] if x['tag'] == t_ref][0]
                mm = leaf_rows[t_mine][th]
                ok &= (rr['rank'], rr['distinct']) == (mm['rank'], mm['distinct'])
                log('    E=%d %s: rank %d distinct %d (nu %d nt %d) | September %s: rank %d distinct %d (nu %d nt %d)'
                    % (th, t_mine, mm['rank'], mm['distinct'], mm['nu'], mm['nt'], t_ref, rr['rank'], rr['distinct'],
                       rr['nu'], rr['nt']))
        log('  validation: max |<r> - September| over E = 3..6, leaves and blocks: %.1e' % dev)
        ok &= dev < 1e-9
        res['validation'] = dict(reference=os.path.relpath(ref_path, HERE), max_r_dev=dev, ok=bool(ok))
        assert ok, 'validation failed'
    res['seconds'] = time.time() - T0
    with open(os.path.join(OUT, 'Hidden_lean_%s_L%d.json' % (direction, TH_MAX)), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, 'Hidden_lean_%s_L%d_values.npz' % (direction, TH_MAX)),
                        **{'E%d__%s' % (th, t): v for t, dd in leaf_dist.items() for th, v in dd.items()})
    log('[total %.0f s]' % (time.time() - T0))
    with open(os.path.join(OUT, 'hidden_lean_%s_L%d.log' % (direction, TH_MAX)), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return res


if __name__ == '__main__':
    for d in sys.argv[1].split(','):
        run(d, check='--check-pairs' in sys.argv, force='--force' in sys.argv)
