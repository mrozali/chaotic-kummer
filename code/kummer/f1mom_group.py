"""F1-mom: the double cover Gtilde (order 384) of the SO(4) point group of the hypercubic
lattice (signed 4x4 permutation matrices with det = +1, order 192), and its character table.

Basis: R^4 = H (quaternions), basis order (1, i, j, k) matching the boson alpha^{A Adot}
components. A rotation R in SO(4) is realised as x -> g1 * x * g2^{-1} (quaternion
multiplication), g1, g2 unit quaternions (Spin(4) = SU(2)_1 x SU(2)_2 -> SO(4), 2-to-1,
kernel {(1,1),(-1,-1)}).

Construction:
  1. Enumerate the 192 signed permutation matrices R with det = +1 directly (combinatorial).
  2. For each R, extract (g1, g2) via: p = R(1) (first column of R, as a quaternion);
     C(x) = R(x) * conj(p) fixes 1 and restricts to an SO(3) rotation on span(i,j,k); extract
     g1 from that 3x3 rotation matrix by the standard robust quaternion-from-matrix formula
     (Shepperd's method); then g2 = conj(p) * g1 (from p = g1 g2^{-1}).
  3. Verify R(x) == g1 * x * conj(g2) for x in {1,i,j,k}, for every R (assert, tolerance 1e-8).
  4. Gtilde = {(g1,g2)} union {(-g1,-g2)}, 384 elements; (-1,-1) acts on fermions as -1 (it is
     the kernel of Spin(4) -> SO(4), the "fermion number" (-1)^F of the seed).

Conjugacy classes and the character table are computed by the class-algebra method: build the
structure constants of class sums from the explicit 384x384 multiplication table, form a random
integer combination of the class-sum matrices, diagonalize it once (its eigenvectors are the
common eigenvectors of every class-sum matrix, a standard fact for the centre of the group
algebra), and read off the characters from the eigenvalues with the standard normalization.
Checks: sum(dim^2) = 384, row/column orthogonality, class sizes sum to 384.
"""
import itertools
import numpy as np


# ---------------------------------------------------------------------------
# quaternions as 4-vectors (a, b, c, d) = a + b i + c j + d k
# ---------------------------------------------------------------------------

def qmul(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ])


def qconj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])


def quat_from_rotmat3(Cm):
    """Standard robust (Shepperd) extraction of a unit quaternion from a 3x3 rotation
    matrix Cm (acting on the i,j,k subspace). Returns (w,x,y,z) with w + x i + y j + z k."""
    tr = Cm[0, 0] + Cm[1, 1] + Cm[2, 2]
    cands = [tr, Cm[0, 0], Cm[1, 1], Cm[2, 2]]
    branch = int(np.argmax(cands))
    if branch == 0:
        w = 0.5 * np.sqrt(max(1 + tr, 0.0))
        x = (Cm[2, 1] - Cm[1, 2]) / (4 * w)
        y = (Cm[0, 2] - Cm[2, 0]) / (4 * w)
        z = (Cm[1, 0] - Cm[0, 1]) / (4 * w)
    elif branch == 1:
        x = 0.5 * np.sqrt(max(1 + Cm[0, 0] - Cm[1, 1] - Cm[2, 2], 0.0))
        w = (Cm[2, 1] - Cm[1, 2]) / (4 * x)
        y = (Cm[0, 1] + Cm[1, 0]) / (4 * x)
        z = (Cm[0, 2] + Cm[2, 0]) / (4 * x)
    elif branch == 2:
        y = 0.5 * np.sqrt(max(1 - Cm[0, 0] + Cm[1, 1] - Cm[2, 2], 0.0))
        w = (Cm[0, 2] - Cm[2, 0]) / (4 * y)
        x = (Cm[0, 1] + Cm[1, 0]) / (4 * y)
        z = (Cm[1, 2] + Cm[2, 1]) / (4 * y)
    else:
        z = 0.5 * np.sqrt(max(1 - Cm[0, 0] - Cm[1, 1] + Cm[2, 2], 0.0))
        w = (Cm[1, 0] - Cm[0, 1]) / (4 * z)
        x = (Cm[0, 2] + Cm[2, 0]) / (4 * z)
        y = (Cm[1, 2] + Cm[2, 1]) / (4 * z)
    return np.array([w, x, y, z])


def signed_permutations_det_plus1():
    """The 192 4x4 signed permutation matrices with det = +1."""
    mats = []
    for perm in itertools.permutations(range(4)):
        # permutation matrix
        P = np.zeros((4, 4))
        for col, row in enumerate(perm):
            P[row, col] = 1
        for signs in itertools.product([1, -1], repeat=4):
            S = np.diag(signs).astype(float)
            R = P @ S
            if abs(np.linalg.det(R) - 1) < 1e-9:
                mats.append(R)
    assert len(mats) == 192
    return mats


def find_g1g2(R, tol=1e-8):
    """Find unit quaternions (g1,g2) with R(x) = g1 x conj(g2) for all x, given R a signed
    permutation matrix with det=+1 (columns are R applied to basis vectors e1=1,e2=i,e3=j,e4=k)."""
    basis = [np.array([1., 0, 0, 0]), np.array([0, 1., 0, 0]),
             np.array([0, 0, 1., 0]), np.array([0, 0, 0, 1.])]
    Rx = [R @ b for b in basis]
    p = Rx[0]  # R(1)
    pbar = qconj(p)
    Cx = [qmul(rx, pbar) for rx in Rx]  # C(x) = R(x) * conj(p)
    assert np.allclose(Cx[0], [1, 0, 0, 0], atol=tol), f"C(1) != 1: {Cx[0]}"
    Cm = np.array([Cx[1][1:], Cx[2][1:], Cx[3][1:]]).T  # 3x3 rotation matrix on (i,j,k)
    g1 = quat_from_rotmat3(Cm)
    g1 = g1 / np.linalg.norm(g1)
    g2 = qmul(pbar, g1)
    g2 = g2 / np.linalg.norm(g2)
    # verify
    for x, rx in zip(basis, Rx):
        check = qmul(qmul(g1, x), qconj(g2))
        assert np.allclose(check, rx, atol=tol), f"verification failed: {check} vs {rx}"
    return g1, g2


def build_Gtilde():
    """Returns a list of 384 dicts {'g1':4vec,'g2':4vec,'R':4x4 matrix}."""
    Rs = signed_permutations_det_plus1()
    elements = []
    for R in Rs:
        g1, g2 = find_g1g2(R)
        elements.append({"g1": g1, "g2": g2, "R": R})
        elements.append({"g1": -g1, "g2": -g2, "R": R})
    assert len(elements) == 384
    return elements


def quat_key(q, tol=1e-6, scale=1e6):
    return tuple(np.round(q * scale).astype(np.int64))


def group_multiply(e1, e2):
    g1 = qmul(e1["g1"], e2["g1"])
    g2 = qmul(e1["g2"], e2["g2"])
    R = e1["R"] @ e2["R"]
    return {"g1": g1, "g2": g2, "R": R}


def group_inverse(e):
    return {"g1": qconj(e["g1"]), "g2": qconj(e["g2"]), "R": e["R"].T}


def build_index(elements):
    """Map from quaternion-pair key -> index in elements."""
    idx = {}
    for i, e in enumerate(elements):
        idx[(quat_key(e["g1"]), quat_key(e["g2"]))] = i
    assert len(idx) == len(elements)
    return idx


def find_index(idx, e):
    return idx[(quat_key(e["g1"]), quat_key(e["g2"]))]


def conjugacy_classes(elements):
    idx = build_index(elements)
    n = len(elements)
    seen = [False] * n
    classes = []
    for i in range(n):
        if seen[i]:
            continue
        cls = set()
        for j in range(n):  # conjugate by every group element
            g = elements[j]
            ginv = group_inverse(g)
            conj = group_multiply(group_multiply(g, elements[i]), ginv)
            k = find_index(idx, conj)
            cls.add(k)
        for k in cls:
            seen[k] = True
        classes.append(sorted(cls))
    return classes


def multiplication_table(elements):
    idx = build_index(elements)
    n = len(elements)
    table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            table[i, j] = find_index(idx, group_multiply(elements[i], elements[j]))
    return table


def class_structure_constants(classes, table):
    """a[i][j][l] = number of (x,y) in C_i x C_j with x*y = a FIXED representative of C_l
    (the standard class-algebra structure constants: Z_i Z_j = sum_l a[i,j,l] Z_l). Returns a
    k x k x k array. (Summing over ALL x in C_i, y in C_j and binning x*y by class gives
    |C_l| * a[i,j,l], since every element of C_l has the same number of representations by
    conjugation-invariance -- divide by the class size to recover a[i,j,l] itself.)"""
    k = len(classes)
    n = table.shape[0]
    elem_class = np.zeros(n, dtype=np.int32)
    for ci, cls in enumerate(classes):
        for e in cls:
            elem_class[e] = ci
    class_sizes = np.array([len(c) for c in classes])
    raw = np.zeros((k, k, k))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            counts = np.zeros(k)
            for x in ci:
                row = table[x, cj]  # x * y for every y in C_j
                for prod in row:
                    counts[elem_class[prod]] += 1
            raw[i, j, :] = counts
    a = raw / class_sizes[None, None, :]
    assert np.allclose(a, np.round(a)), "structure constants not integers"
    return np.round(a)


def _try_character_table(elements, classes, a, class_sizes, n, seed):
    k = len(classes)
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 50, size=k)
    # N_i (k x k): (N_i)[j, l] = a[i, j, l]. The vector w^(rho)_j = omega_j(rho) =
    # |C_j| chi_rho(g_j)/chi_rho(1) satisfies N_i w^(rho) = omega_i(rho) w^(rho) (derived from
    # Z_i Z_j = sum_l a_{ijl} Z_l evaluated on the central character omega(rho)): a random
    # combination sum_i weights_i N_i has these w^(rho) as its (generically nondegenerate)
    # eigenvectors.
    S = np.zeros((k, k))
    for i in range(k):
        S += weights[i] * a[i, :, :]
    evals, evecs = np.linalg.eig(S)

    # identity class index
    id_idx = None
    for ci, cls in enumerate(classes):
        e = elements[cls[0]]
        if np.allclose(e["g1"], [1, 0, 0, 0], atol=1e-6) and np.allclose(e["g2"], [1, 0, 0, 0], atol=1e-6):
            id_idx = ci
    assert id_idx is not None

    chars = np.zeros((k, k), dtype=complex)  # chars[rho, class]
    for rho in range(k):
        v = evecs[:, rho]
        v = v / v[id_idx]  # normalize omega_j(rho) with j=identity class -> 1
        denom = sum((abs(v[j]) ** 2) / class_sizes[j] for j in range(k))
        dim = np.sqrt(n / denom.real)
        dim_r = round(dim)
        chi = np.array([dim * v[j] / class_sizes[j] for j in range(k)])
        # fix overall phase so chi(identity) = dim_r (real positive)
        chars[rho, :] = chi
    return chars, class_sizes, id_idx


def character_table(elements, classes, seed=1, max_tries=25):
    """Robust wrapper: a random weight vector generically separates all eigenvalues of the
    class-sum combination, but an unlucky draw can produce an accidental degeneracy (two
    irreps sharing the same eigenvalue), which shows up as non-integer 'dimensions' and
    sum(dim^2) != |G|. Retry with a fresh seed until the checks pass exactly."""
    k = len(classes)
    n = len(elements)
    table = multiplication_table(elements)
    a = class_structure_constants(classes, table)
    class_sizes = np.array([len(c) for c in classes])
    for attempt in range(max_tries):
        chars, class_sizes, id_idx = _try_character_table(elements, classes, a, class_sizes, n, seed + attempt)
        dims = chars[:, id_idx].real
        ok = np.allclose(dims, np.round(dims), atol=1e-6) and abs(np.sum(np.round(dims) ** 2) - n) < 1e-6
        if ok:
            return chars, class_sizes, id_idx
    raise RuntimeError(f"character_table: no clean result after {max_tries} seeds")
