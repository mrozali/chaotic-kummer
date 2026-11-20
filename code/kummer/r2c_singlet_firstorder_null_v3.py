"""Singlet first-order gap ratio: chiral GOE reference matched to the deduplicated data.

Replaces the nt x rank reference of r2c_singlet_firstorder_null_v2.py (not included in this repository).
The singlet's resolved
sub-blocks have exact degeneracies, and the data count each degenerate value once, so nt x rank matrices pooled more
gap ratios than the data (159, 1414, 4079, 10661 against 138, 1283, 3788, 10078 at E = 5..8) and gave too narrow a
spread.  Here every fully resolved sub-block with >= 20 distinct values (resolution B of Singlet_firstorder_L8.npz,
columns of B_meta: [4] E, [5] nu, [6] nt, [7] rank, [8] distinct) gets a real Gaussian matrix with nt rows and as many
columns as the sub-block has distinct non-zero singular values, so that the reference pools exactly as many gap ratios
as the data.  Dense SVD, gap ratios per block, pooled; mean and sd of <r> over the samples.  Seeds
numpy.random.default_rng([20261105, E]); 1000 samples at E = 5..7, 300 at E = 8.  Along the generic directions, which
have no exact degeneracies, nt x distinct is nt x rank, so their reference is unchanged.

Usage: python r2c_singlet_firstorder_null_v3.py npz out.json [levels] [nsamp_E<=7] [nsamp_E8]
  levels: comma-separated, default 5,6,7,8; an existing out.json is updated level by level, so the levels can be run
  separately (E = 8 takes about 10 min on two cores).
"""
import json
import os
import sys
import time

import numpy as np

SEED = 20261105


def gap_ratios(d):
    s = np.diff(np.sort(d))
    s = s[s > 0]
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])


def level(meta, E, ns):
    blocks = [(int(m[6]), int(m[7]), int(m[8])) for m in meta if m[4] == E and m[8] >= 20]   # nt, rank, distinct
    rng = np.random.default_rng([SEED, E])
    means, t0 = [], time.time()
    for _ in range(ns):
        g = np.concatenate([gap_ratios(np.linalg.svd(rng.standard_normal((nt, d)), compute_uv=False))
                            for nt, _, d in blocks])
        means.append(float(g.mean()))
    means = np.array(means)
    return dict(n_blocks=len(blocks), blocks_nt_rank_distinct=blocks,
                data_ratios=int(sum(d - 2 for _, _, d in blocks)),
                old_nt_x_rank_ratios=int(sum(r - 2 for _, r, _ in blocks)),
                nt_x_distinct=dict(nsamp=ns, n_ratios=int(g.size), mean=float(means.mean()),
                                   sd=float(means.std(ddof=1)), se_of_mean=float(means.std(ddof=1) / np.sqrt(ns)),
                                   sd_rel_precision=float(1 / np.sqrt(2 * (ns - 1))),
                                   seconds=round(time.time() - t0, 1), means=means.tolist()))


def main(npz, out, levels="5,6,7,8", n_lo=1000, n_8=300):
    meta = np.load(npz, allow_pickle=False)["B_meta"]
    res = json.load(open(out)) if os.path.exists(out) else dict(doc=__doc__, seed=SEED, levels={})
    for E in (int(x) for x in str(levels).split(",")):
        lev = level(meta, E, int(n_8) if E == 8 else int(n_lo))
        r = lev["nt_x_distinct"]
        print("E=%d  %4d samples  %5d ratios (data %5d, nt x rank %5d)  mean %.4f  sd %.4f  [%.0f s]"
              % (E, r["nsamp"], r["n_ratios"], lev["data_ratios"], lev["old_nt_x_rank_ratios"], r["mean"], r["sd"],
                 r["seconds"]), flush=True)
        res["levels"][str(E)] = lev
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
