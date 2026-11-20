"""r2g_generic_probe.py -- first-order in-block statistics for a GENERIC direction of the
blow-up deformation.

The four Z_2^4-symmetric blow-up moduli are G^-_{Adot,-1/2} Gbar^-_{Bdot,-1/2} sigma^{++}; on a
spin-0 block the untwisted -> twisted first-order matrix of a direction c is

    M_c = kappa * sum_{Ad,Bd} c[Ad,Bd]  A_Ad (x) A_Bd ,

the paper's deformation being the SU(2)_2 singlet c = eps (c[PLUS,MINUS] = -1, c[MINUS,PLUS] = +1,
the "wedge").  Symmetric c is the triplet.  For each direction and level th (E = th, spin 0):

  1. raw block M_c(th): twisted raw (nW^2) <- untwisted raw (du), from Space.A;
  2. stabiliser H_c = {g in Gtilde : U_t(g) M_c = s_g M_c U_u(g), s_g = +-1}, found NUMERICALLY
     at a small level (it is a property of c, not of the level); s_g = -1 elements are
     symmetries only together with the grading G = diag(+1 untwisted, -1 twisted);
  3. character table of H_c (f1mom_group on the sub-list), isotypic projectors on both sides,
     physical irreps only (chi(z) = dim, z the central orbifold Z_2, always in H_c);
  4. the extra Z_2's among Pi (L<->R exchange), Z_R, F_R (right-only central elements), each
     possibly times G, split jointly inside every isotypic component;
  5. singular values per sub-block; each multiplicity-space value appears dim(sigma) times, so
     distinct values (unique to 9 digits, as in r2c_gate11) are the multiplicity-space spectrum;
     gap ratios on sub-blocks with >= 20 distinct values, pooled per level.

Validation: c = eps must reproduce the per-irrep, unsplit data
(d_+, d_-, rank, zero modes) and the Pi G x Z_R split result (<r> = 0.456 +- 0.025
at E = 5, 20 sub-blocks, 353 distinct values).

Usage:  python r2g_generic_probe.py <direction> <th_max> [th_min]
        direction in {singlet, trip0, trip+, trip-, generic, generic_c}
Writes r2_out/Generic_probe_<direction>_L<th_max>.json and appends to r2_out/Generic_probe.md.
"""
import json
import os
import sys
import time

import numpy as np

import f1mom_group as fg
import f2_common as f2
import r2_d0 as d0
import r2c_group as gr
import r2c_space as sp
from r2_algebra import PLUS, MINUS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_GUE, R_POI = 0.5307, 0.5996, 0.3863


# ----------------------------------------------------------------------------- directions
def cmat(name):
    c = np.zeros((2, 2), complex)
    if name == 'singlet':
        c[PLUS, MINUS] = -1.0
        c[MINUS, PLUS] = 1.0
    elif name == 'trip0':
        c[PLUS, MINUS] = 1.0
        c[MINUS, PLUS] = 1.0
    elif name == 'trip+':
        c[PLUS, PLUS] = np.sqrt(2.0)
    elif name == 'trip-':
        c[MINUS, MINUS] = np.sqrt(2.0)
    elif name == 'generic':
        rng = np.random.default_rng(2026)
        t = rng.standard_normal((2, 2))
        t = (t + t.T) / 2
        t *= np.sqrt(2.0) / np.linalg.norm(t)
        c = (cmat('singlet') + t) / np.sqrt(2.0)
    elif name == 'generic_c':
        rng = np.random.default_rng(7)
        t = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        c = t * np.sqrt(2.0) / np.linalg.norm(t)
    # ---- PHYSICAL directions ----------------
    # Hermiticity of D_c = c^{AB} eps eps G Gbar sigma requires c^* = eps c eps^T up to an overall
    # phase, i.e. c a QUATERNION: c = a 1 + i (b sigma_x + c sigma_y + d sigma_z), a,b,c,d real.
    # The singlet is eps = -i sigma_y; the other three quaternion units span the triplet.
    # 'trip0' = sigma_x = i (-i sigma_x) is physical up to the (unitarily trivial) overall
    # phase; 'trip+', 'generic', 'generic_c' are NOT first-order matrices of any local modulus.
    elif name == 'trip1':          # identity, a triplet direction
        c[PLUS, PLUS] = 1.0
        c[MINUS, MINUS] = 1.0
    elif name == 'tripz':          # i sigma_z ~ sigma_z (phase), a triplet direction
        c[PLUS, PLUS] = 1.0
        c[MINUS, MINUS] = -1.0
    elif name == 'q45':            # singlet + identity, equal weights (real matrix)
        c = (cmat('singlet') + cmat('trip1')) / np.sqrt(2.0)
    elif name == 'qgen':           # generic unit quaternion: all four components
        rng = np.random.default_rng(2026)
        a, b, cc, d = rng.standard_normal(4)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        c = a * np.eye(2) + 1j * (b * sx + cc * sy + d * sz)
        c *= np.sqrt(2.0) / np.linalg.norm(c)
    elif name == 'qgen2':          # a second generic unit quaternion
        rng = np.random.default_rng(99)
        a, b, cc, d = rng.standard_normal(4)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        c = a * np.eye(2) + 1j * (b * sx + cc * sy + d * sz)
        c *= np.sqrt(2.0) / np.linalg.norm(c)
    elif name.startswith('qrand'):     # random unit quaternion from an integer seed, e.g. qrand101 (direction scan)
        rng = np.random.default_rng(int(name[5:]))
        a, b, cc, d = rng.standard_normal(4)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        c = a * np.eye(2) + 1j * (b * sx + cc * sy + d * sz)
        c *= np.sqrt(2.0) / np.linalg.norm(c)
    elif name.startswith('plane'):     # a qrand direction projected EXACTLY onto a coordinate plane: plane<seed>_<comp>, comp in abcd
        seed, comp = name[5:].split('_')
        rng = np.random.default_rng(int(seed))
        q = rng.standard_normal(4)
        q['abcd'.index(comp)] = 0.0
        a, b, cc, d = q
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        c = a * np.eye(2) + 1j * (b * sx + cc * sy + d * sz)
        c *= np.sqrt(2.0) / np.linalg.norm(c)
    elif name.startswith('wall'):      # random direction constrained to a candidate wall: wall<seed>_<spec>, spec in
        # a0 b0 c0 d0 (one component zero), or two letters with optional 'm' for a minus sign: bc (b=c), bmc (b=-c), ...
        seed, spec = name[4:].split('_')
        rng = np.random.default_rng(int(seed))
        q = rng.standard_normal(4)
        idx = 'abcd'
        if spec == 'ctrl':               # control plane, normal (1,2,3,4)/sqrt(30), not expected to be a wall
            nrm = np.array([1., 2., 3., 4.]) / np.sqrt(30.)
            q = q - (nrm @ q) * nrm
        elif len(spec) == 4 and set(spec) <= set('pm'):   # F4 plane: sum_i s_i q_i = 0 with s from 'p'/'m'
            nrm = np.array([1. if ch == 'p' else -1. for ch in spec]) / 2.
            q = q - (nrm @ q) * nrm
        elif spec[1] == '0':
            q[idx.index(spec[0])] = 0.0
        else:
            i, j = idx.index(spec[0]), idx.index(spec[-1])
            q[j] = -q[i] if 'm' in spec else q[i]
        a, b, cc, d = q
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        sz = np.array([[1, 0], [0, -1]], complex)
        c = a * np.eye(2) + 1j * (b * sx + cc * sy + d * sz)
        c *= np.sqrt(2.0) / np.linalg.norm(c)
    else:
        raise ValueError(name)
    return c


def is_physical(c, tol=1e-9):
    """True if c is a quaternion up to an overall phase: c^* = e^{i phi} eps c eps^T."""
    eps = np.array([[0, -1], [1, 0]], complex)
    lhs = c.conj()
    rhs = eps @ c @ eps.T
    i = np.unravel_index(np.argmax(np.abs(rhs)), rhs.shape)
    if abs(rhs[i]) < tol:
        return False
    ph = lhs[i] / rhs[i]
    return abs(abs(ph) - 1) < tol and np.allclose(lhs, ph * rhs, atol=tol)


# ----------------------------------------------------------------------------- raw blocks
def M_c_raw(S, th, c):
    """kappa sum_{Ad,Bd} c[Ad,Bd] A_Ad (x) A_Bd : raw untwisted level th -> twisted level th."""
    A = {Ad: S.A[th].get((th, Ad)) for Ad in (PLUS, MINUS)}
    nW = S.nW[th]
    off, tot = S.sect_offsets(th)
    M = np.zeros((nW * nW, tot), complex)
    if A[PLUS] is None or A[MINUS] is None:
        return M
    for s in S.sect[th]:
        a, b = s
        oa, ob = S.voff[th][a], S.voff[th][b]
        na, nb = S.vdim[th][a], S.vdim[th][b]
        blk = np.zeros((nW * nW, na * nb), complex)
        for Ad in (PLUS, MINUS):
            for Bd in (PLUS, MINUS):
                if abs(c[Ad, Bd]) < 1e-14:
                    continue
                blk += c[Ad, Bd] * np.kron(A[Ad][:, oa:oa + na], A[Bd][:, ob:ob + nb])
        M[:, off[s]:off[s] + blk.shape[1]] = d0.KAPPA * blk
    return M


def U_tw_raw(S, th, gi):
    U = S.Utw[th][gi]
    return np.kron(U, U)


def raw_extra_ops(S, th):
    """Pi (exchange of the chiral factors), Z_R and F_R (right-only central elements) on the
    raw spin-0 spaces at level th: {name: (Xu, Xt)}."""
    g = S.g
    off, du = S.sect_offsets(th)
    nW = S.nW[th]
    Pu = np.zeros((du, du))
    for (a, b) in S.sect[th]:
        na, nb = S.vdim[th][a], S.vdim[th][b]
        o_ab, o_ba = off[(a, b)], off[(b, a)]
        i = np.repeat(np.arange(na), nb)
        j = np.tile(np.arange(nb), na)
        Pu[o_ba + j * na + i, o_ab + i * nb + j] = 1.0
    Pt = np.zeros((nW * nW, nW * nW))
    i = np.repeat(np.arange(nW), nW)
    j = np.tile(np.arange(nW), nW)
    Pt[j * nW + i, i * nW + j] = 1.0
    ops = {'Pi': (Pu.astype(complex), Pt.astype(complex))}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xu = np.zeros((du, du), complex)
        for (a, b) in S.sect[th]:
            na, nb = S.vdim[th][a], S.vdim[th][b]
            bp, Mb = S.Vmat(b, gi, th)
            tgt = (a, bp)
            blk = np.kron(np.eye(na), Mb)
            Xu[off[tgt]:off[tgt] + blk.shape[0], off[(a, b)]:off[(a, b)] + blk.shape[1]] = blk
        Xt = np.kron(np.eye(nW), S.Utw[th][gi])
        ops[nm] = (Xu, Xt)
    return ops


def classify_ops(ops, M, tol=1e-9):
    """Which candidates commute (+1) or anticommute (-1, then X G is the symmetry) with M."""
    nrm = max(float(np.max(np.abs(M))), 1e-30)
    res = {}
    for nm, (Xu, Xt) in ops.items():
        cp = float(np.max(np.abs(Xt @ M - M @ Xu))) / nrm
        cm = float(np.max(np.abs(Xt @ M + M @ Xu))) / nrm
        if cp < tol:
            res[nm] = (Xu, Xt, +1, cp)
        elif cm < tol:
            res[nm] = (Xu, -Xt, -1, cm)
        else:
            res[nm] = (None, None, 0, min(cp, cm))
    return res


# ----------------------------------------------------------------------------- stabiliser
def stabiliser(S, c, th_test):
    """H_c and the sign s_g, by direct test at level th_test; also the analytic rule."""
    g = S.g
    M = M_c_raw(S, th_test, c)
    nrm = float(np.max(np.abs(M)))
    Hc, sgn = [], {}
    rule_hits = {'Yb c Yb^T': 0, 'Yb^T c Yb': 0, 'Yb^* c Yb^+': 0}
    for gi in range(g.n):
        Uu = S.U_untw_raw(th_test, gi)
        Ut = U_tw_raw(S, th_test, gi)
        X, Y = Ut @ M, M @ Uu
        s = 0
        if np.max(np.abs(X - Y)) < 1e-9 * nrm:
            s = +1
        elif np.max(np.abs(X + Y)) < 1e-9 * nrm:
            s = -1
        if s:
            Hc.append(gi)
            sgn[gi] = s
            Yb = g.Yb[gi]
            for nm, cc in (('Yb c Yb^T', Yb @ c @ Yb.T), ('Yb^T c Yb', Yb.T @ c @ Yb),
                           ('Yb^* c Yb^+', Yb.conj() @ c @ Yb.conj().T)):
                if np.allclose(cc, s * c, atol=1e-9):
                    rule_hits[nm] += 1
    # closure and multiplicativity of the sign
    Hs = set(Hc)
    for i in Hc:
        for j in Hc:
            k = g.mult(i, j)
            assert k in Hs, 'stabiliser not closed'
            assert sgn[k] == sgn[i] * sgn[j], 'sign not multiplicative'
    return Hc, sgn, rule_hits


def subgroup_characters(g, Hc):
    """Character table of the subgroup H_c (indices into Hc), physical filter, class map."""
    els = [g.els[i] for i in Hc]
    classes = fg.conjugacy_classes(els)
    chars, class_sizes, id_idx = fg.character_table(els, classes)
    elem_class = np.zeros(len(els), int)
    for ci, cl in enumerate(classes):
        for e in cl:
            elem_class[e] = ci
    dims = np.rint(chars[:, id_idx].real).astype(int)
    iz = Hc.index(g.i_z)
    izF = Hc.index(g.i_zF)
    cz, czF = elem_class[iz], elem_class[izF]
    # physical (z-even) AND bosonic (zF-even): the only irreps that occur on spin-0 RR blocks
    phys = np.array([abs(chars[r, cz].real - dims[r]) < 1e-6 and
                     abs(chars[r, czF].real - dims[r]) < 1e-6 for r in range(len(dims))])
    return chars, dims, elem_class, phys, len(classes)


# ----------------------------------------------------------------------------- resolution
def isotypic_bases(S, th, Hc, sgn, chars, elem_class, dims, phys, log):
    du = S.sect_offsets(th)[1]
    nW = S.nW[th]
    dt = nW * nW
    irr = [r for r in range(len(dims)) if phys[r]]
    Pu = {r: np.zeros((du, du), complex) for r in irr}
    Pt = {r: np.zeros((dt, dt), complex) for r in irr}
    nH = len(Hc)
    t0 = time.time()
    for k, gi in enumerate(Hc):
        Uu = S.U_untw_raw(th, gi)
        Ut = sgn[gi] * U_tw_raw(S, th, gi)
        cl = elem_class[k]
        for r in irr:
            w = dims[r] / nH * np.conj(chars[r, cl])
            if abs(w) < 1e-15:
                continue
            Pu[r] += w * Uu
            Pt[r] += w * Ut
    log('    projectors accumulated over %d elements in %.1f s' % (nH, time.time() - t0))
    bases, diag = {}, {}
    worst = 0.0
    rng = np.random.default_rng(5)
    for r in irr:
        for tag, P in (('u', Pu[r]), ('t', Pt[r])):
            herm = float(np.max(np.abs(P - P.conj().T)))
            m = int(np.rint(np.trace(P).real))
            if m == 0:
                Q = np.zeros((P.shape[0], 0), complex)
            else:
                # randomised range finder: range(P) has dimension Tr P exactly
                X = rng.standard_normal((P.shape[0], m + 8)) + 1j * rng.standard_normal((P.shape[0], m + 8))
                Y = P @ X
                Qf, R = np.linalg.qr(Y)
                keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])
                Q = Qf[:, keep]
                assert Q.shape[1] == m, (r, tag, Q.shape[1], m)
                idem = float(np.max(np.abs(P @ Q - Q)))
                worst = max(worst, idem)
            worst = max(worst, herm)
            bases[(r, tag)] = Q
            diag[(r, tag)] = Q.shape[1]
        Pu[r] = None
        Pt[r] = None
    tot_u = sum(diag[(r, 'u')] for r in irr)
    tot_t = sum(diag[(r, 't')] for r in irr)
    log('    |P-P^+|, |P^2-P| worst %.1e ; physical dims: untwisted %d of %d raw, twisted %d of %d raw'
        % (worst, tot_u, du, tot_t, dt))
    return bases, diag, irr


def split_joint(Qu, Qt, syms, names, rng):
    """Joint eigenspaces of the extra Z_2's inside one isotypic component."""
    if not names:
        return {(): (Qu, Qt)}
    coef = rng.standard_normal(len(names))
    out = {}
    for tag, Q in (('u', Qu), ('t', Qt)):
        if Q.shape[1] == 0:
            continue
        Ys = []
        for nm in names:
            X = syms[nm][0] if tag == 'u' else syms[nm][1]
            Y = Q.conj().T @ X @ Q
            Ys.append((Y + Y.conj().T) / 2)
        Z = sum(cf * Y for cf, Y in zip(coef, Ys))
        w, V = np.linalg.eigh(Z)
        labs = []
        for i in range(V.shape[1]):
            v = V[:, i]
            lab = []
            for Y in Ys:
                e = float(np.real(np.conj(v) @ (Y @ v)))
                assert abs(abs(e) - 1) < 1e-6, e
                lab.append(int(np.sign(e)))
            labs.append(tuple(lab))
        for lab in set(labs):
            cols = [i for i, l in enumerate(labs) if l == lab]
            out.setdefault(lab, {})[tag] = Q @ V[:, cols]
    res = {}
    for lab, d in out.items():
        qu = d.get('u', np.zeros((Qu.shape[0], 0), complex))
        qt = d.get('t', np.zeros((Qt.shape[0], 0), complex))
        res[lab] = (qu, qt)
    return res


# ----------------------------------------------------------------------------- main
def run(direction, th_max, th_min=2):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    c = cmat(direction)
    log('# Generic-direction probe: %s, th <= %d  (%s)' % (direction, th_max,
                                                          time.strftime('%Y-%m-%d %H:%M')))
    log('c = %s' % np.array2string(c, precision=4))
    log('PHYSICAL (quaternion up to phase, i.e. a local Hermitian modulus): %s' % is_physical(c))
    t0 = time.time()
    S = sp.Space(th_max, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built in %.1f s' % (th_max, time.time() - t0))

    th_test = min(3, th_max)
    Hc, sgn, rule = stabiliser(S, c, th_test)
    n_minus = sum(1 for v in sgn.values() if v < 0)
    log('stabiliser H_c: order %d of 384 (%d elements act with G); analytic rule hits: %s'
        % (len(Hc), n_minus, rule))
    chars, dims, elem_class, phys, ncls = subgroup_characters(g, Hc)
    log('H_c has %d classes, %d irreps of dims %s, %d physical (chi(z) = dim)'
        % (ncls, len(dims), sorted(dims.tolist()), int(phys.sum())))

    results = {'direction': direction, 'c': [[str(x) for x in row] for row in c],
               'H_order': len(Hc), 'H_minus': n_minus, 'irreps': int(len(dims)),
               'phys': int(phys.sum()), 'levels': {}}
    rng = np.random.default_rng(17)
    for th in range(th_min, th_max + 1):
        if th not in S.nW or S.nW[th] == 0:
            continue
        t1 = time.time()
        M = M_c_raw(S, th, c)
        du = M.shape[1]
        dt = M.shape[0]
        gmax = float(np.max(np.abs(M)))
        log('\n## E = %d : raw untwisted %d, twisted %d' % (th, du, dt))
        # extra Z_2's
        ops = raw_extra_ops(S, th)
        cls = classify_ops(ops, M)
        syms = {nm: (v[0], v[1]) for nm, v in cls.items() if v[2] != 0}
        names = sorted(syms)
        log('    extra Z_2 candidates: ' + ', '.join('%s:%s' % (nm, {1: 'commutes', -1: 'anticommutes (x G)',
                                                                       0: 'no'}[v[2]]) for nm, v in cls.items()))
        bases, diag, irr = isotypic_bases(S, th, Hc, sgn, chars, elem_class, dims, phys, log)
        # per-irrep (unsplit) table and joint split
        rows = []
        gaps = []
        n_sub = 0
        n_distinct = 0
        tot_rank = tot_dp = tot_dm = 0
        raw_sv_all = []
        for r in irr:
            Qu, Qt = bases[(r, 'u')], bases[(r, 't')]
            d = int(dims[r])
            if Qu.shape[1] == 0 or Qt.shape[1] == 0:
                continue
            Mr = Qt.conj().T @ M @ Qu
            sv = np.linalg.svd(Mr, compute_uv=False) if Mr.size else np.zeros(0)
            nz = sv[sv > 1e-9 * gmax]
            raw_sv_all += list(nz)
            dp, dm = Qu.shape[1] // d, Qt.shape[1] // d
            rank = len(nz) / d
            rows.append(dict(rho=int(r), dim=d, d_plus=dp, d_minus=dm, rank=rank,
                             zero_modes=dp + dm - 2 * rank,
                             n_distinct=int(len(np.unique(np.round(nz, 9))))))
            tot_rank += rank
            tot_dp += dp
            tot_dm += dm
            # joint split by the extra Z_2's
            parts = split_joint(Qu, Qt, syms, names, rng)
            for lab, (qu, qt) in sorted(parts.items()):
                if qu.shape[1] == 0 or qt.shape[1] == 0:
                    continue
                Ms = qt.conj().T @ M @ qu
                s2 = np.linalg.svd(Ms, compute_uv=False)
                s2 = s2[s2 > 1e-9 * gmax]
                dist = np.unique(np.round(s2, 9))
                n_sub += 1
                n_distinct += len(dist)
                if len(dist) >= 20:
                    gaps.append(f2.gap_ratios(dist))
        gaps = np.concatenate(gaps) if gaps else np.zeros(0)
        rmean = float(np.mean(gaps)) if len(gaps) else float('nan')
        rsem = float(np.std(gaps) / np.sqrt(len(gaps))) if len(gaps) else float('nan')
        log('    multiplicity space: d_+ %d, d_- %d, rank %.1f, zero modes %.1f (fraction %.2f), '
            'bipartite bound %d' % (tot_dp, tot_dm, tot_rank, tot_dp + tot_dm - 2 * tot_rank,
                                    (tot_dp + tot_dm - 2 * tot_rank) / max(tot_dp + tot_dm, 1),
                                    abs(tot_dp - tot_dm)))
        log('    resolved sub-blocks %d (irreps x %s), distinct values %d, gap ratios %d : <r> = %.3f +- %.3f'
            '   [GOE %.3f, GUE %.3f, Poisson %.3f]' % (n_sub, names, n_distinct, len(gaps), rmean, rsem,
                                                        R_GOE, R_GUE, R_POI))
        log('    | rho | dim | d_+ | d_- | rank | zero modes | # distinct |')
        for x in rows:
            log('    | %d | %d | %d | %d | %g | %g | %d |' % (x['rho'], x['dim'], x['d_plus'], x['d_minus'],
                                                          x['rank'], x['zero_modes'], x['n_distinct']))
        # completeness check against the raw SVD where affordable
        comp = None
        if du <= 3000:
            ref = np.linalg.svd(M, compute_uv=False)
            ref = ref[ref > 1e-9 * gmax]
            allsv = np.sort(np.array(raw_sv_all))[::-1]
            n = min(len(ref), len(allsv))
            comp = (int(len(ref)), int(len(allsv)),
                    float(np.max(np.abs(ref[:n] - allsv[:n]))) if n else 0.0)
            log('    completeness: raw rank %d vs physical isotypic rank %d, max |dsv| %.1e'
                ' (unphysical z-odd part carries the rest)' % comp)
        results['levels'][th] = dict(du=du, dt=dt, d_plus=tot_dp, d_minus=tot_dm, rank=tot_rank,
                                     zero_modes=tot_dp + tot_dm - 2 * tot_rank, n_sub=n_sub,
                                     n_distinct=n_distinct, n_gaps=int(len(gaps)), r_mean=rmean,
                                     r_sem=rsem, extra_Z2={nm: int(v[2]) for nm, v in cls.items()},
                                     rows=rows, completeness=comp, seconds=time.time() - t1)
        log('    [%.1f s]' % (time.time() - t1))
    fn = os.path.join(OUT, 'Generic_probe_%s_L%d.json' % (direction, th_max))
    with open(fn, 'w') as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(os.path.join(OUT, 'Generic_probe.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('\nwritten %s' % fn)
    return results


if __name__ == '__main__':
    direction = sys.argv[1] if len(sys.argv) > 1 else 'singlet'
    th_max = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    th_min = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    run(direction, th_max, th_min)
