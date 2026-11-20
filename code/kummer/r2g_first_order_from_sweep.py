"""r2g_first_order_from_sweep.py -- exact first-order statistics LEVEL BY LEVEL from the saved sweep
matrices of r2g_generic_sweep.py, after the paper's hidden-symmetry resolution (r2c_gate19_lib.fully_resolve).

The sweep saves, per (irrep, extra-Z2 labels) block, the bare levels E0 (all th <= th_max, each basis
vector with a definite bare level) and the full level-changing D_0 block M.  The level-diagonal part
M[Et == th, Eu == th] is EXACTLY the first-order matrix of degenerate perturbation theory at E = th, so
its singular values are the first-order spectrum, resolved by every symmetry the sweep resolves PLUS
the hidden Z_2's that fully_resolve finds (which act level by level: level_plusminus_basis).

Outputs (r2_out/):
  Generic_leaves_<dir>_L<th_max>.npz         the fully resolved leaves (E0_k, M_k, nu_k, tag_k) -- reused if present
  Generic_firstorder_<dir>_L<th_max>.json    per-level table (leaves and unsplit blocks), KS distances, T7 check
  Generic_firstorder_<dir>_L<th_max>.npz     per-level, per-leaf distinct first-order values (for P(r) plots)
  Generic_firstorder.md                      appended log

Usage: python r2g_first_order_from_sweep.py <direction> <th_max> [--force-leaves]
"""
import json
import os
import sys
import time

import numpy as np

import f2_common as f2
import r2c_gate19_lib as L
from r2c_build import SubSector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_GUE, R_POI = 0.5307, 0.5996, 0.3863


# ------------------------------------------------------------------ reference distributions of the folded gap ratio
def goe_surmise_cdf(r):
    """Folded (r <= 1) Atas et al. surmise for beta = 1, P(r) = 2 (27/8)(r + r^2)/(1 + r + r^2)^{5/2}."""
    x = np.linspace(0, 1, 20001)
    p = 2 * (27 / 8) * (x + x ** 2) / (1 + x + x ** 2) ** 2.5
    c = np.concatenate([[0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(x))])
    c /= c[-1]
    return np.interp(r, x, c)


def poisson_cdf(r):
    return 2 * r / (1 + r)


def ks(gaps, cdf):
    g = np.sort(gaps)
    n = len(g)
    if n == 0:
        return float('nan')
    F = cdf(g)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n)))


# ------------------------------------------------------------------ leaves
def get_leaves(direction, th_max, log, force=False):
    path = os.path.join(OUT, 'Generic_leaves_%s_L%d.npz' % (direction, th_max))
    if os.path.exists(path) and not force:
        d = np.load(path, allow_pickle=False)
        n = int(d['n_leaves'])
        leaves = [(str(d['tag_%d' % k]), SubSector(d['E0_%d' % k], d['M_%d' % k], int(d['nu_%d' % k])))
                  for k in range(n)]
        log('loaded %d leaves from %s' % (n, os.path.basename(path)))
        return leaves, None
    d = np.load(os.path.join(OUT, 'Generic_sweep_%s_L%d_matrices.npz' % (direction, th_max)))
    keys = list(d['keys'])
    blocks, leaves = [], []
    for k, key in enumerate(keys):
        sub = SubSector(d['E0_%d' % k], d['M_%d' % k], int(d['nu_%d' % k][0]))
        blocks.append((str(key), sub))
        t1 = time.time()
        lg = []
        lv = L.fully_resolve(sub, th_max, tag='b%d' % k, log=lg)
        for tag, info, n in lg:
            if isinstance(info, dict):
                log('  %s (n=%d): components %d sizes %s, singles %d, unlifted %d, thresh 1e%.0f, stable %s'
                    % (tag, n, info['n_components'], info['comp_sizes'][:6], info['n_singles'],
                       info['n_unlifted'], np.log10(max(info['thresh_used'], 1e-300)), info['stable']))
            else:
                log('  %s: %s (n=%d)' % (tag, info, n))
        log('block %d %s: n=%d -> %d leaves of sizes %s  [%.0f s]'
            % (k, key, sub.n, len(lv), [x[1].n for x in lv], time.time() - t1))
        leaves += [('%s|%s' % (key, tag), s) for tag, s, depth in lv]
    npz = dict(n_leaves=np.array(len(leaves)))
    for k, (tag, s) in enumerate(leaves):
        npz['tag_%d' % k] = np.array(tag)
        npz['E0_%d' % k] = s.E0
        npz['M_%d' % k] = s.M
        npz['nu_%d' % k] = np.array(s.nu)
    np.savez_compressed(path, **npz)
    log('saved %d leaves to %s' % (len(leaves), os.path.basename(path)))
    return leaves, blocks


# ------------------------------------------------------------------ first order, level by level
def level_table(named_subs, levels, gmax, log, label):
    """For each level th: per sub-block the level-diagonal singular values; pooled gap ratios."""
    table, dist_store = {}, {}
    for th in levels:
        rows, gaps = [], []
        tot_u = tot_t = tot_rank = tot_dist = 0
        for k, (tag, s) in enumerate(named_subs):
            iu = s.E0[:s.nu] == th
            it = s.E0[s.nu:] == th
            nu_th, nt_th = int(iu.sum()), int(it.sum())
            if nu_th == 0 or nt_th == 0:
                if nu_th or nt_th:
                    rows.append(dict(sub=k, tag=tag, nu=nu_th, nt=nt_th, rank=0, distinct=0))
                    tot_u += nu_th
                    tot_t += nt_th
                continue
            Ms = s.M[it][:, iu]
            sv = np.linalg.svd(Ms, compute_uv=False)
            sv = sv[sv > 1e-9 * gmax]
            dist = np.unique(np.round(sv, 9))
            rows.append(dict(sub=k, tag=tag, nu=nu_th, nt=nt_th, rank=int(len(sv)), distinct=int(len(dist))))
            dist_store[(th, k)] = dist
            tot_u += nu_th
            tot_t += nt_th
            tot_rank += len(sv)
            tot_dist += len(dist)
            if len(dist) >= 20:
                gaps.append(f2.gap_ratios(dist))
        gaps = np.concatenate(gaps) if gaps else np.zeros(0)
        rm = float(np.mean(gaps)) if len(gaps) else float('nan')
        se = float(np.std(gaps) / np.sqrt(len(gaps))) if len(gaps) else float('nan')
        table[th] = dict(d_plus=tot_u, d_minus=tot_t, rank=tot_rank, distinct=tot_dist, n_gaps=int(len(gaps)),
                         r_mean=rm, r_sem=se, ks_goe=ks(gaps, goe_surmise_cdf), ks_poisson=ks(gaps, poisson_cdf),
                         rows=rows)
        log('  [%s] E=%d: d+ %d, d- %d, rank %d, distinct %d, gaps %d : <r> = %.4f +- %.4f  (KS GOE %.3f, Poisson %.3f)'
            % (label, th, tot_u, tot_t, tot_rank, tot_dist, len(gaps), rm, se, table[th]['ks_goe'],
               table[th]['ks_poisson']))
        for x in rows:
            log('      | %d | %s | %d | %d | %d | %d |' % (x['sub'], x['tag'], x['nu'], x['nt'], x['rank'], x['distinct']))
    return table, dist_store


# ---------------------------------------------------------------- T7: small-lambda clusters of H(lambda) vs first order
def t7_check(leaves, levels, gmax, log, lam=None):
    """At lam << lam*, the eigenvalues of the full truncated H(lam) near bare level th are
    th + lam (+-s_i) + O(lam^2) with s_i the first-order singular values; compare cluster by cluster."""
    out = {}
    smax = 0.0
    for tag, s in leaves:
        for th in levels:
            iu = s.E0[:s.nu] == th
            it = s.E0[s.nu:] == th
            if iu.any() and it.any():
                sv = np.linalg.svd(s.M[it][:, iu], compute_uv=False)
                smax = max(smax, float(sv[0]))
    if lam is None:
        lam = min(0.002, 0.15 / max(smax, 1e-12))
    log('  T7 at lambda = %.5f (largest first-order singular value %.2f, cluster half-width %.3f)' % (lam, smax, lam * smax))
    for th in levels:
        dev_max, dev_rel, n_cl, n_fo, spacing = 0.0, 0.0, 0, 0, []
        for tag, s in leaves:
            iu = s.E0[:s.nu] == th
            it = s.E0[s.nu:] == th
            n_th = int(iu.sum() + it.sum())
            if n_th == 0:
                continue
            ev = np.linalg.eigvalsh(s.H(lam))
            cl = ev[np.abs(ev - th) < 0.5]
            fo = np.zeros(n_th)
            if iu.any() and it.any():
                sv = np.linalg.svd(s.M[it][:, iu], compute_uv=False)
                sv = sv[sv > 1e-9 * gmax]
                fo[:len(sv)] = sv
                fo[len(sv):2 * len(sv)] = -sv
                spacing.append(np.median(np.diff(np.unique(np.round(sv, 9)))) if len(sv) > 2 else np.nan)
            fo = np.sort(fo)
            n_cl += len(cl)
            n_fo += n_th
            if len(cl) == n_th:
                x = (np.sort(cl) - th) / lam
                dev = np.max(np.abs(x - fo))
                dev_max = max(dev_max, dev)
        sp = float(np.nanmedian(spacing)) if spacing else float('nan')
        out[th] = dict(n_cluster=n_cl, n_states=n_fo, max_dev_over_lam=dev_max, median_spacing=sp,
                       max_dev_in_spacings=dev_max / sp if sp else float('nan'))
        log('  T7 E=%d: cluster states %d of %d, max |(E - th)/lam - (+-s)| = %.4f = %.3f median first-order spacings'
            % (th, n_cl, n_fo, dev_max, dev_max / sp if sp else float('nan')))
    return out, lam


# ------------------------------------------------------------------ main
def main(direction, th_max, force=False):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    T0 = time.time()
    log('# First-order statistics level by level from the sweep: %s, th_max %d  [%s]'
        % (direction, th_max, time.strftime('%Y-%m-%d %H:%M')))
    leaves, blocks = get_leaves(direction, th_max, log, force)
    if blocks is None:
        d = np.load(os.path.join(OUT, 'Generic_sweep_%s_L%d_matrices.npz' % (direction, th_max)))
        blocks = [(str(key), SubSector(d['E0_%d' % k], d['M_%d' % k], int(d['nu_%d' % k][0])))
                  for k, key in enumerate(list(d['keys']))]
    levels = sorted(set(int(x) for tag, s in leaves for x in s.E0))
    gmax = max(float(np.max(np.abs(s.M))) for tag, s in blocks)
    log('%d blocks -> %d leaves; levels %s; max |M| %.3f' % (len(blocks), len(leaves), levels, gmax))
    log('## unsplit blocks (the sweep resolution)')
    tab_b, _ = level_table(blocks, levels, gmax, log, 'blocks')
    log('## fully resolved leaves')
    tab_l, dist = level_table(leaves, levels, gmax, log, 'leaves')
    log('## summary: E | <r> blocks | <r> leaves | gaps (leaves) | KS GOE | KS Poisson')
    for th in levels:
        b, l = tab_b[th], tab_l[th]
        log('  E=%d | %.4f +- %.4f | %.4f +- %.4f | %d | %.3f | %.3f'
            % (th, b['r_mean'], b['r_sem'], l['r_mean'], l['r_sem'], l['n_gaps'], l['ks_goe'], l['ks_poisson']))
    log('  reference: GOE %.4f, GUE %.4f, Poisson %.4f' % (R_GOE, R_GUE, R_POI))
    log('## T7: small-coupling clusters of the truncated Hamiltonian against the first-order values (leaves)')
    t7, lam7 = t7_check(leaves, [th for th in levels if th >= th_max - 1], gmax, log)
    res = dict(direction=direction, th_max=th_max, n_blocks=len(blocks), n_leaves=len(leaves),
               leaf_sizes=[s.n for tag, s in leaves], leaf_tags=[tag for tag, s in leaves], levels=levels,
               blocks={str(k): v for k, v in tab_b.items()}, leaves={str(k): v for k, v in tab_l.items()},
               t7=dict(lam=lam7, levels={str(k): v for k, v in t7.items()}), seconds=time.time() - T0)
    with open(os.path.join(OUT, 'Generic_firstorder_%s_L%d.json' % (direction, th_max)), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, 'Generic_firstorder_%s_L%d.npz' % (direction, th_max)),
                        **{'E%d_leaf%d' % (th, k): v for (th, k), v in dist.items()})
    with open(os.path.join(OUT, 'Generic_firstorder.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.0f s]' % (time.time() - T0))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), force='--force-leaves' in sys.argv)
