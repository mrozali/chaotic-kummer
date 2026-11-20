"""Route 2 Phase A, GATE 6: the first-order deformation operator D_0 on the relative theory.

    D_0 = kappa * sum_{Adot Bdot} eps^{Adot Bdot} [ T_Adot (x) Tbar_Bdot ]_{h - hbar conserved}

Why only sigma^{++}.  The full deformation is
D = (1/4) eps^{Adot Bdot} eps_{alpha beta} eps_{albar betabar} G^{alpha}_{Adot,-1/2}
    Gbar^{albar}_{Bdot,-1/2} sigma^{beta betabar}, and the twist field factorises into a left
and a right spin field, so the chiral object is S_Adot = G^+_{Adot,-1/2} sigma^-
- G^-_{Adot,-1/2} sigma^+.  The SU(2)_L triplet partner G^+_{Adot,-1/2} sigma^+ VANISHES
(sigma^+ is a chiral primary: 1912.05567 eq B.5, verified here as the exact intertwining
identity G^{+,tw}_{Adot,n} Sigma = Sigma G^{+(g)}_{Adot,n}), hence
G^+_{Adot,-1/2} sigma^- = - G^-_{Adot,-1/2} sigma^+ (their eq B.6, verified numerically in
r2_gate6, not included in this repository) and S_Adot = -2 T_Adot.  Therefore
D = eps^{Adot Bdot} T_Adot (x) Tbar_Bdot exactly,
with no extra terms, and the 1/4 is cancelled by the two factors of -2.

Overall normalisation kappa.  D_0 = int_0^{2pi} dsigma D(sigma, tau=0).  At w_0 = i sigma the
level-graded amplitude carries z_0^{h_t-h_u} zbar_0^{hbar_t-hbar_u} = e^{i sigma (spin change)},
so the integral is 2 pi times the spin-conserving part of D(w_0 = 0).  With D UNIT NORMALISED
(<D(z,zbar) D(0)> = 1/|z|^4) the cylinder two-point function is
<D_0(tau) D_0(0)> = pi^2 cosh(tau)/sinh^3(tau) = sum_n 4 pi^2 n^2 e^{-2 n tau},
i.e. sum over twisted states at E = 2n of |<t|D_0|0>|^2 = 4 pi^2 n^2.  For the untwisted R
ground state, T_Adot|0> = G^{-,tw}_{Adot,0}|chi> and the level-n components A_Adot^{(n)}
satisfy (measured in r2_gate6, not included in this repository) ||A_+^{(n)}||^2 = ||A_-^{(n)}||^2 = n exactly and
<A_+|A_-> = 0, so the left-hand side is kappa^2 (||A_+||^2 ||A_-||^2 + ||A_-||^2 ||A_+||^2)
= 2 kappa^2 n^2.  Hence kappa^2 = 2 pi^2:

    kappa = sqrt(2) pi = 2 pi / sqrt(2)

i.e. the operator eps^{Adot Bdot} G^-_{Adot,-1/2} Gbar^-_{Bdot,-1/2} sigma^{++} used by
Guo-Mathur has two-point normalisation 2, not 1, and the unit-normalised deformation is
1/sqrt2 times it.  (The level-independence of the extracted kappa -- ||A^{(n)}||^2 = n for
every n tested -- is the actual content of the check.)

Hermiticity.  The twisted -> untwisted block is the adjoint of the untwisted -> twisted one,
which is 1912.05567 eq (B.4) combined with (B.6); the machinery here implements only the
untwisted -> twisted twist map, so Hermiticity is IMPOSED, and the block matrix is
[[0, M^dag],[M, 0]] whose spectrum is +/- the singular values of M.
"""
import numpy as np

import r2_algebra as al
import r2_lift as lf
import r2_rel as rl
from r2_algebra import UNTW, TW, PLUS, MINUS, two_h

KAPPA = np.sqrt(2.0) * np.pi


def coupling_parity(th):
    """The Z_2 parity of the untwisted relative chiral states that the level-preserving
    twist map does NOT annihilate.  Sigma commutes with the orbifold Z_2, which acts as
    (-1)^{#relative operators} on the untwisted side and as (-1)^{two_h} on the twisted
    side (each twisted relative mode is half-odd, so the half-shift gives -1)."""
    return 1 if th % 2 == 0 else -1


def chiral_A(th, prim_u=None, prim_t=None, Ads=(PLUS, MINUS)):
    """Matrices of T_Adot between the ORTHONORMAL massive-primary bases of the relative
    theory at two_h = th.  Only the coupling parity is used on the untwisted side.
    Returns (A, prim_u, prim_t) with A[Adot] of shape (len(prim_t), len(prim_u))."""
    par = coupling_parity(th)
    if prim_u is None:
        prim_u = rl.massive_primaries(UNTW, th, parity=par)
    if prim_t is None:
        prim_t = rl.massive_primaries(TW, th)
    A = {}
    for Ad in Ads:
        M = np.zeros((len(prim_t), len(prim_u)), dtype=complex)
        for j, v in enumerate(prim_u):
            im = lf.T_Adot(v, Ad, th=th)
            if not im:
                continue
            for i, q in enumerate(prim_t):
                z = TW.ip(q, im)
                if abs(z) > 1e-13:
                    M[i, j] = z
        A[Ad] = M
    return A, prim_u, prim_t


def block_M(Ap_L, Am_L, Ap_R=None, Am_R=None, kappa=KAPPA):
    """The untwisted -> twisted block of D_0 on a spin-0 block, in the tensor-product
    primary bases:  M = kappa ( A_-^L (x) A_+^R  -  A_+^L (x) A_-^R ).
    (eps^{+-} = -1, eps^{-+} = +1.)"""
    if Ap_R is None:
        Ap_R = Ap_L
    if Am_R is None:
        Am_R = Am_L
    return kappa * (np.kron(Am_L, Ap_R) - np.kron(Ap_L, Am_R))


def block_report(M, d_plus, d_minus, tol=None):
    """Singular values, rank, zero-mode count, Tr D_B^2 and the bandwidth of the Hermitian
    block matrix [[0, M^dag],[M, 0]] on a space of d_plus untwisted + d_minus twisted states
    (M may have fewer columns than d_plus when part of the untwisted space is annihilated
    for parity reasons: the missing columns are exactly zero)."""
    if M.size == 0:
        sv = np.zeros(0)
    else:
        sv = np.linalg.svd(M, compute_uv=False)
    if tol is None:
        tol = 1e-10 * max(sv[0], 1.0) if len(sv) else 1.0
    rank = int(np.sum(sv > tol))
    nz = d_plus + d_minus - 2 * rank
    return dict(sv=sv, rank=rank, zero_modes=nz, bound=abs(d_plus - d_minus),
                trD2=2.0 * float(np.sum(sv ** 2)), sigma_max=float(sv[0]) if len(sv) else 0.0,
                bandwidth=2.0 * (float(sv[0]) if len(sv) else 0.0))
