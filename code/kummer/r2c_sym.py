"""Route 2 Phase C: the symmetries of H(lambda) BEYOND Gtilde.

Two exact Z_2's survive the deformation and must be resolved before any level statistics
(an unresolved Z_2 superposes two independent spectra and biases every statistic towards
Poisson):

  Pi   : exchange of the left and right chiral factors (worldsheet parity).  It satisfies
         Pi_t D_0 Pi_u = -D_0, because every column of D_0 is the WEDGE
         A_- (x) A_+ - A_+ (x) A_-.  Combined with the grading G = +1 (untwisted),
         -1 (twisted) -- the quantum Z_2 of the orbifold, under which the twist field is
         odd -- the product  Lambda = Pi G  commutes with H(lambda) for every lambda.
  Z_R  : the orbifold Z_2 acting on the RIGHT movers only, i.e. the central Gtilde element
         z = (-1,+1) applied to the right chiral factor alone.  Because Yb(z) = 1 the
         SU(2)_2 index of T_Adot is not rotated, so the chiral covariance gives
         U_W(z) A_Adot = A_Adot U(z) and Z_R commutes with D_0 exactly.  On physical
         (z-even) states Z_L = Z_R, so this is a single extra Z_2.
  F_R  : the same construction with zF = (-1,-1) = (-1)^F.  Yb(zF) = -1, so it
         ANTI-commutes with D_0 and the symmetry is F_R G.  On physical states
         F_L = F_R, and F_R Z_R is the right-only (+1,-1).

The group generated is (Z_2)^k; the code below finds which of the candidates actually
commute and splits the multiplicity space accordingly.
"""
import numpy as np

import r2c_group as gr


def side_op(sec, S, gi, side):
    """Matrices of the LEFT-only (side='L') or RIGHT-only (side='R') action of the Gtilde
    element gi on the multiplicity basis of `sec`.  Returns (Xu, Xt)."""
    nu, nt = sec.nu, sec.nt
    Xu = np.zeros((nu, nu), complex)
    for th in sorted(sec.mu):
        off = sec.uoff[th]
        B = sec.Bu[th]
        for (a, b), (c0, blk) in B.items():
            if side == 'R':
                bp, Mb = S.Vmat(b, gi, th)
                tgt = (a, bp)
                if tgt not in B:
                    continue
                t = np.tensordot(blk, Mb, axes=([1], [1]))        # (p, a, q)
                t = np.transpose(t, (0, 2, 1))                    # (p, q, a)
            else:
                ap, Ma = S.Vmat(a, gi, th)
                tgt = (ap, b)
                if tgt not in B:
                    continue
                t = np.tensordot(Ma, blk, axes=([1], [0]))        # (p, q, a)
            c1, blk2 = B[tgt]
            Xu[off + c1:off + c1 + blk2.shape[2],
               off + c0:off + c0 + blk.shape[2]] += np.tensordot(np.conj(blk2), t,
                                                                 axes=([0, 1], [0, 1]))
    Xt = np.zeros((nt, nt), complex)
    for th in sorted(sec.mt):
        off = sec.toff[th]
        B = sec.Bt[th]
        m = B.shape[2]
        U = S.Utw[th][gi]
        if side == 'R':
            t = np.transpose(np.tensordot(B, U, axes=([1], [1])), (0, 2, 1))
        else:
            t = np.tensordot(U, B, axes=([1], [0]))
        Xt[off:off + m, off:off + m] = np.tensordot(np.conj(B), t, axes=([0, 1], [0, 1]))
    return Xu, Xt


def candidates(sec, S):
    """{name: (Xu, Xt, sign)} for every candidate involution, with sign = +1 if it commutes
    with D_0 and -1 if it anticommutes (in which case the symmetry is X times the grading
    G = diag(+1 untwisted, -1 twisted))."""
    g = gr.G()
    out = {}
    Pu, Pt = sec.parity_operator()
    out['Pi'] = (Pu, Pt)
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        out[nm] = side_op(sec, S, gi, 'R')
    return out


def exact_symmetries(sec, S, tol=1e-9):
    """The exact Z_2 symmetries of H(lambda) beyond Gtilde, as (Xu, Xt) pairs already
    including the grading factor where needed.  Returns {name: (Xu, Xt)} with
    X = diag(Xu, Xt) commuting with H(lambda) for every lambda."""
    nrm = max(float(np.max(np.abs(sec.M))), 1e-30)
    out = {}
    info = {}
    for nm, (Xu, Xt) in candidates(sec, S).items():
        cp = float(np.max(np.abs(Xt @ sec.M - sec.M @ Xu))) / nrm
        cm = float(np.max(np.abs(Xt @ sec.M + sec.M @ Xu))) / nrm
        trivial = (np.max(np.abs(Xu - np.eye(sec.nu))) < 1e-9 and
                   np.max(np.abs(Xt + np.eye(sec.nt))) < 1e-9) or \
                  (np.max(np.abs(Xu - np.eye(sec.nu))) < 1e-9 and
                   np.max(np.abs(Xt - np.eye(sec.nt))) < 1e-9)
        info[nm] = (cp, cm, trivial)
        if trivial:
            continue
        if cp < tol:
            out[nm] = (Xu, Xt)
        elif cm < tol:
            out[nm] = (Xu, -Xt)          # X times the grading G
    return out, info


def split_all(sec, S, tol=1e-6):
    """Split the Gtilde multiplicity space of `sec` into the joint eigenspaces of every
    exact Z_2 beyond Gtilde.  Returns (names, {label: SubSector})."""
    from r2c_build import SubSector
    syms, info = exact_symmetries(sec, S)
    names = sorted(syms)
    rng = np.random.default_rng(17)
    coef = rng.standard_normal(len(names)) if names else np.zeros(0)
    parts = {}
    for tag, off, mm, lev in (('u', sec.uoff, sec.mu, sec.ulev),
                              ('t', sec.toff, sec.mt, sec.tlev)):
        for th in lev:
            o, m = off[th], mm[th]
            ops = [(syms[nm][0] if tag == 'u' else syms[nm][1])[o:o + m, o:o + m]
                   for nm in names]
            if not ops:
                V = np.eye(m)
                labs = [()] * m
            else:
                Z = sum(c * (Y + Y.conj().T) / 2 for c, Y in zip(coef, ops))
                w, V = np.linalg.eigh(Z)
                labs = []
                for i in range(m):
                    v = V[:, i]
                    lab = tuple(int(np.sign(np.real(np.conj(v) @ (Y @ v)))) for Y in ops)
                    for Y in ops:
                        e = np.real(np.conj(v) @ (Y @ v))
                        assert abs(abs(e) - 1) < tol, e
                    labs.append(lab)
            for i in range(m):
                full = np.zeros(sec.nu if tag == 'u' else sec.nt, complex)
                full[o:o + m] = V[:, i]
                parts.setdefault(labs[i], {}).setdefault(tag, []).append((float(th), full))
    out = {}
    for lab, d in parts.items():
        qu = d.get('u', [])
        qt = d.get('t', [])
        Qu = np.stack([x[1] for x in qu], axis=1) if qu else np.zeros((sec.nu, 0))
        Qt = np.stack([x[1] for x in qt], axis=1) if qt else np.zeros((sec.nt, 0))
        Eu = np.array([x[0] for x in qu])
        Et = np.array([x[0] for x in qt])
        ou = np.argsort(Eu, kind='stable') if len(Eu) else np.zeros(0, int)
        ot = np.argsort(Et, kind='stable') if len(Et) else np.zeros(0, int)
        Qu, Eu = Qu[:, ou], Eu[ou]
        Qt, Et = Qt[:, ot], Et[ot]
        M = np.conj(Qt.T) @ sec.M @ Qu
        out[lab] = SubSector(np.concatenate([Eu, Et]), M, len(Eu))
    return names, out


def classify(sec, S, tol=1e-9):
    """Which candidates are exact symmetries; returns {name: (Xu, Xt, grade)} with
    grade = +1 (X itself commutes) or -1 (X G commutes)."""
    nrm = max(float(np.max(np.abs(sec.M))), 1e-30)
    res = {}
    for nm, (Xu, Xt) in candidates(sec, S).items():
        cp = float(np.max(np.abs(Xt @ sec.M - sec.M @ Xu))) / nrm
        cm = float(np.max(np.abs(Xt @ sec.M + sec.M @ Xu))) / nrm
        inv = float(np.max(np.abs(Xu @ Xu - np.eye(sec.nu))))
        if cp < tol:
            res[nm] = (Xu, Xt, +1, inv, cp)
        elif cm < tol:
            res[nm] = (Xu, Xt, -1, inv, cm)
    return res
