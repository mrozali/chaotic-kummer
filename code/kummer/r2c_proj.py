"""Route 2 Phase C: Gtilde isotypic projectors and multiplicity ("weight e_1 slice") bases.

For a representation U of Gtilde on a space X the operators
    P^rho_{ij} = (d_rho/|G|) sum_g conj(D^rho_{ij}(g)) U(g)
satisfy P_{ij}P_{kl} = delta_{jk} P_{il}, P_{ij}^dag = P_{ji}, sum_{rho} sum_i P^rho_{ii} = 1.
range(P^rho_{11}) is the "e_1 component" of the rho-isotypic part; its dimension is the
multiplicity m_rho, and any operator commuting with U (H(lambda) does) preserves it and acts
on it as the multiplicity-space operator H_rho.  Eigenvalues obtained there are the honest
non-degenerate levels: the full isotypic component repeats each of them d_rho times.
"""
import numpy as np

import r2c_group as gr
import r2c_irreps as ir


def proj_matrix(U, D, i, j):
    """P^rho_{ij} as a dense matrix, U = {gi: matrix}, D = (n_g, d, d) irrep array."""
    n = next(iter(U.values())).shape[0]
    d = D.shape[1]
    P = np.zeros((n, n), complex)
    for gi, M in U.items():
        c = np.conj(D[gi][i, j])
        if abs(c) > 1e-14:
            P += c * M
    return P * (d / len(U))


def irrep_adapt(U, D, tol=1e-8):
    """Orthonormal irrep-adapted vectors: returns Q of shape (n, m, d) with
    Q[:, alpha, i] = |rho, alpha, i>, or None if the multiplicity is zero."""
    P11 = proj_matrix(U, D, 0, 0)
    w, V = np.linalg.eigh((P11 + P11.conj().T) / 2)
    keep = w > 0.5
    m = int(np.sum(keep))
    if m == 0:
        return None
    Q1 = V[:, keep]
    d = D.shape[1]
    Q = np.zeros((Q1.shape[0], m, d), complex)
    Q[:, :, 0] = Q1
    for i in range(1, d):
        Q[:, :, i] = proj_matrix(U, D, i, 0) @ Q1
    return Q


def cg_slice(Ds, Dt, Dr):
    """Weight-1 slices of the Dr-isotypic part of Ds (x) Dt: array (N, ds*dt)."""
    ds, dt, dr = Ds.shape[1], Dt.shape[1], Dr.shape[1]
    ng = Ds.shape[0]
    P = np.zeros((ds * dt, ds * dt), complex)
    for gi in range(ng):
        c = np.conj(Dr[gi][0, 0])
        if abs(c) > 1e-14:
            P += c * np.kron(Ds[gi], Dt[gi])
    P *= dr / ng
    w, V = np.linalg.eigh((P + P.conj().T) / 2)
    keep = w > 0.5
    if not np.any(keep):
        return np.zeros((0, ds * dt), complex)
    return V[:, keep].T


def tensor_mult_basis(U, rho, adapt_cache=None):
    """Multiplicity basis of irrep `rho` in X (x) X with the DIAGONAL Gtilde action, where
    U = {gi: matrix on X}.  Returns a list of (n x n) coefficient matrices."""
    I = ir.I()
    n = next(iter(U.values())).shape[0]
    if adapt_cache is None:
        adapt_cache = {}
    Qs = {}
    for s in range(I.nirr):
        if s not in adapt_cache:
            adapt_cache[s] = irrep_adapt(U, I.D[s])
        if adapt_cache[s] is not None:
            Qs[s] = adapt_cache[s]
    out = []
    for s, Qsig in Qs.items():
        for t, Qtau in Qs.items():
            C = cg_slice(I.D[s], I.D[t], I.D[rho])
            if not len(C):
                continue
            ds, dt = I.D[s].shape[1], I.D[t].shape[1]
            ms, mt = Qsig.shape[1], Qtau.shape[1]
            Cr = C.reshape(-1, ds, dt)
            for a in range(ms):
                for b in range(mt):
                    # v = sum_ij c_ij q^s_{a i} (q^t_{b j})^T
                    T = np.einsum('nij,pi,qj->npq', Cr, Qsig[:, a, :], Qtau[:, b, :])
                    for k in range(T.shape[0]):
                        out.append(T[k])
    return out


def hom_basis(rhoW_pairs, Dr, n_a, n_b, tol=1e-8, seed=3):
    """Basis of Hom_K(Res_K rho, V_a (x) V_b): phi with shape (n_a, n_b, d) satisfying
    (Ma(k) (x) Mb(k)) phi = phi rho(k).  `rhoW_pairs` = list of (Ma, Mb, Dk) over k in K."""
    d = Dr.shape[1]
    K = len(rhoW_pairs)
    # exact multiplicity from characters
    m = 0.0
    for (Ma, Mb, Dk) in rhoW_pairs:
        m += np.conj(np.trace(Dk)) * np.trace(Ma) * np.trace(Mb)
    m = int(np.rint((m / K).real))
    if m == 0:
        return np.zeros((0, n_a, n_b, d), complex)
    rng = np.random.default_rng(seed)
    nv = m + 6
    phi = (rng.standard_normal((nv, n_a, n_b, d))
           + 1j * rng.standard_normal((nv, n_a, n_b, d)))
    acc = np.zeros_like(phi)
    for (Ma, Mb, Dk) in rhoW_pairs:
        t = np.einsum('pi,nijc->npjc', Ma, phi)
        t = np.einsum('qj,npjc->npqc', Mb, t)
        acc += np.einsum('npqc,ec->npqe', t, np.conj(Dk))
    acc /= K
    X = acc.reshape(nv, -1)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(np.sum(s > tol * max(s[0], 1e-30)))
    assert r == m, (r, m)
    return Vt[:m].reshape(m, n_a, n_b, d)
