"""Explicit unitary matrix irreps D^rho(g) of Gtilde, extracted from the regular
representation: project on the rho-isotypic component (dimension d^2), average a random
Hermitian matrix over the group to get an element of the commutant (= m (x) 1_d), and take
one eigenspace of it -- an invariant subspace carrying a single copy of rho."""
import numpy as np
import r2c_group as gr


class Irreps:
    def __init__(self, tol=1e-7, seed=7):
        g = gr.G()
        self.g = g
        n = g.n
        self.n = n
        self.dims = g.dims
        self.nirr = len(self.dims)
        self.D = []
        tab = g.table
        rng = np.random.default_rng(seed)
        for r in range(self.nirr):
            d = int(self.dims[r])
            chi = g.chars[r][g.elem_class]
            # ---- isotypic projector in the regular rep, applied to the whole space -----
            P = np.zeros((n, n), complex)
            for a in range(n):
                P[tab[a], np.arange(n)] += np.conj(chi[a])
            P *= d / n
            U, s, Vt = np.linalg.svd(P)
            rk = int(np.sum(s > tol * s[0]))
            assert rk == d * d, (r, d, rk)
            Q = U[:, :rk]                                   # n x d^2 orthonormal
            # ---- left action restricted to the isotypic component ---------------------
            Lm = np.zeros((n, rk, rk), complex)
            Qc = np.conj(Q.T)
            for a in range(n):
                Lm[a] = Qc @ Q[tab[a], :] if False else Qc @ Q[np.argsort(tab[a]), :]
            # R(a) e_x = e_{a x}: (R(a) Q)[y, :] = Q[a^{-1} y, :]
            inv = g.inv
            for a in range(n):
                Lm[a] = Qc @ Q[tab[int(inv[a])], :]
            # ---- commutant element ----------------------------------------------------
            Z = rng.standard_normal((rk, rk)) + 1j * rng.standard_normal((rk, rk))
            Z = Z + Z.conj().T
            M = np.zeros((rk, rk), complex)
            for a in range(n):
                M += Lm[a].conj().T @ Z @ Lm[a]
            M /= n
            w, V = np.linalg.eigh(M)
            # group eigenvalues into d clusters of size d
            groups = []
            cur = [0]
            for i in range(1, rk):
                if abs(w[i] - w[cur[-1]]) < 1e-6 * max(1.0, abs(w[i])):
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            gsel = [gg for gg in groups if len(gg) == d]
            assert gsel, (r, d, [len(x) for x in groups])
            sub = V[:, gsel[0]]                             # rk x d, invariant
            E = (Q @ sub).T                                 # d x n orthonormal rows
            Dm = np.zeros((n, d, d), complex)
            for a in range(n):
                Dm[a] = np.conj(E) @ E[:, tab[int(inv[a])]].T
            self.D.append(Dm)
        self._check()

    def _check(self):
        g = self.g
        rng = np.random.default_rng(11)
        for r in range(self.nirr):
            d = int(self.dims[r])
            D = self.D[r]
            for _ in range(8):
                a, b = [int(x) for x in rng.integers(0, self.n, 2)]
                assert np.allclose(D[a] @ D[b], D[g.mult(a, b)], atol=1e-7), ('hom', r)
                assert np.allclose(D[a] @ D[a].conj().T, np.eye(d), atol=1e-7), ('unit', r)
            chi = np.array([np.trace(D[a]) for a in range(self.n)])
            assert np.allclose(chi, g.chars[r][g.elem_class], atol=1e-6), ('char', r)


_I = None


def I():
    global _I
    if _I is None:
        _I = Irreps()
    return _I
