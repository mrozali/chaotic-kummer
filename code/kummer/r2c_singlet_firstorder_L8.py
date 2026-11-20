"""First-order spectra along the SU(2)_2 singlet direction, E = 1..8, from the pre-built
L8 matrices (r2_out/PhaseC4_matrices_prim_L8.npz).  Resolution A = 40 sub-sectors,
Resolution B = fully resolved (hidden Z_2 S), same per-level extraction as r2c_gate11 fo.
Writes Singlet_firstorder_L8.{npz,json,md}.  Gates G1-G4 are checked before
the final files are written (diagnostics go to Singlet_firstorder_L8_GATEFAIL.json if any of
G1,G3,G4 fails)."""
import os, json, time
import numpy as np
import r2c_build as bd
import r2c_gate19_lib as L
import f2_common as f2

OUT = L.OUT
TH = 8
ES = list(range(1, 9))
DIM = {0: 1, 2: 1, 3: 3, 5: 3, 6: 2, 8: 3, 10: 3, 14: 3, 16: 3, 19: 6}
PAPER_RANK = {2: 24, 3: 91, 4: 278, 5: 1434, 6: 5307}
EXPECT_SPLIT = {1: [384, 148], 3: [539, 376], 5: [407, 213], 7: [464, 204], 9: [1305, 607],
                11: [1336, 718], 13: [1346, 719], 15: [1271, 586], 17: [790, 356],
                19: [1011, 553]}
TOL = 1e-9


def block_sv(sub, E):
    Eu, Et = sub.E0[:sub.nu], sub.E0[sub.nu:]
    iu = np.where(Eu == E)[0]
    it = np.where(Et == E)[0]
    if not len(iu) or not len(it):
        return None
    sv = np.linalg.svd(sub.M[np.ix_(it, iu)], compute_uv=False)  # descending, incl zeros
    return sv, len(iu), len(it)


def extract(sv):
    nz = sv[sv > 1e-9 * max(sv[0], 1e-30)]
    d = np.unique(np.round(nz, 9))
    return len(nz), d


def summarize(blocks):
    """blocks: list of (dweight, sv, nuE, ntE). Returns summary dict (reference procedure)."""
    nb = nd = 0
    rs = []
    wrank = 0
    for dw, sv, nuE, ntE in blocks:
        r, d = extract(sv)
        nb += 1
        nd += len(d)
        wrank += dw * r
        if len(d) >= 20:
            rs.append(f2.gap_ratios(d))
    rs = np.concatenate(rs) if rs else np.array([np.nan])
    n = int(np.sum(np.isfinite(rs)))
    return dict(blocks=nb, distinct=int(nd), ratios=n if rs.size and np.isfinite(rs).any() else 0,
                r_mean=float(np.nanmean(rs)) if n else None,
                r_sem=float(np.nanstd(rs) / np.sqrt(max(len(rs), 1))) if n else None,
                weighted_rank=int(wrank))


def main():
    t0 = time.time()
    d = np.load(os.path.join(OUT, 'PhaseC4_matrices_prim_L8.npz'))
    meta = d['meta']
    n40 = meta.shape[0]
    subs = [bd.SubSector(d['E0_%d' % k], d['M_%d' % k], int(meta[k, 5])) for k in range(n40)]
    for k in range(n40):
        assert int(meta[k, 1]) == DIM[int(meta[k, 0])], (k, meta[k])
    print('loaded', n40, flush=True)

    # ---------------- Resolution A
    A_blocks = {E: [] for E in ES}
    A_data, A_rows = {}, []
    for k in range(n40):
        rho = int(meta[k, 0])
        for E in ES:
            r = block_sv(subs[k], E)
            if r is None:
                continue
            sv, nuE, ntE = r
            rk, dist = extract(sv)
            A_data['A_k%d_E%d' % (k, E)] = sv
            A_rows.append((k, rho, int(meta[k, 2]), int(meta[k, 3]), E, nuE, ntE, rk, len(dist)))
            A_blocks[E].append((DIM[rho], sv, nuE, ntE))
    A_sum = {E: summarize(A_blocks[E]) for E in ES}
    for E in ES:
        print('A E=%d' % E, A_sum[E], flush=True)

    # ---------------- G1, G2
    g1 = {}
    s5, s6 = A_sum[5], A_sum[6]
    g1['E5'] = dict(blocks=s5['blocks'], distinct=s5['distinct'], ratios=s5['ratios'],
                    r=s5['r_mean'], sem=s5['r_sem'])
    g1['E6'] = dict(blocks=s6['blocks'], distinct=s6['distinct'], ratios=s6['ratios'],
                    r=s6['r_mean'], sem=s6['r_sem'])
    ok5 = (s5['blocks'] == 20 and s5['distinct'] == 353 and s5['ratios'] == 138 and
           round(s5['r_mean'], 3) == 0.456 and round(s5['r_sem'], 3) == 0.025)
    ok6 = (s6['blocks'] == 20 and s6['distinct'] == 1400 and
           round(s6['r_mean'], 3) == 0.465 and round(s6['r_sem'], 3) == 0.008)
    g1['pass'] = bool(ok5 and ok6)
    g2 = {E: dict(weighted_rank=A_sum[E]['weighted_rank'], paper=PAPER_RANK[E],
                  match=A_sum[E]['weighted_rank'] == PAPER_RANK[E]) for E in PAPER_RANK}
    print('G1', g1, '\nG2', g2, flush=True)

    # ---------------- G3
    g3 = {'pass': True, 'detail': []}
    for rho in (0, 2):
        z = np.load(os.path.join(OUT, 'PhaseC4_matrices_prim_L9_rho%d.npz' % rho))
        m9 = z['meta']
        for j in range(m9.shape[0]):
            k8 = [k for k in range(n40) if tuple(meta[k, :4]) == tuple(m9[j, :4])]
            assert len(k8) == 1, (rho, j, m9[j])
            k8 = k8[0]
            s9 = bd.SubSector(z['E0_%d' % j], z['M_%d' % j], int(m9[j, 5]))
            for E in (7, 8):
                a, b = block_sv(subs[k8], E), block_sv(s9, E)
                if a is None or b is None:
                    ok, dev, note = (a is None and b is None), None, 'empty L8=%s L9=%s' % (a is None, b is None)
                elif a[0].shape != b[0].shape:
                    ok, dev, note = False, None, 'shape %s vs %s' % (a[0].shape, b[0].shape)
                else:
                    dev = float(np.max(np.abs(a[0] - b[0])) / max(a[0].max(), 1e-30))
                    ok, note = dev <= TOL, ''
                g3['pass'] &= bool(ok)
                g3['detail'].append(dict(rho=rho, j=j, k8=k8, E=E, rel_dev=dev, ok=bool(ok), note=note))
                print('G3 rho=%d j=%d k8=%d E=%d dev=%s %s' % (rho, j, k8, E, dev, note), flush=True)

    # ---------------- Resolution B
    leaves = []
    split_ks = []
    sizes = {}
    for k in range(n40):
        t1 = time.time()
        sl = L.fully_resolve(subs[k], TH, tag='k%d' % k, depth=0, max_depth=6, log=[])
        if len(sl) > 1:
            split_ks.append(k)
        sizes[k] = [s.n for _, s, _ in sl]
        for tag, sub, depth in sl:
            leaves.append((k, tag, sub, depth))
        print('B k=%d n=%d -> %s (%.0fs)' % (k, subs[k].n, sizes[k], time.time() - t1), flush=True)

    B_blocks = {E: [] for E in ES}
    B_data, B_rows, B_tags = {}, [], []
    leaf_sv = {}
    for k, tag, sub, depth in leaves:
        rho = int(meta[k, 0])
        ftag = tag.replace('+', 'p').replace('-', 'm')
        suffix = tag[len('k%d' % k):]
        sgn = 0 if not suffix else (1 if suffix[-1] == '+' else -1)
        if ftag not in B_tags:
            B_tags.append(ftag)
        for E in ES:
            r = block_sv(sub, E)
            if r is None:
                continue
            sv, nuE, ntE = r
            rk, dist = extract(sv)
            B_data['B_%s_E%d' % (ftag, E)] = sv
            leaf_sv[(tag, E)] = sv
            B_rows.append((k, rho, int(meta[k, 2]), int(meta[k, 3]), E, nuE, ntE, rk, len(dist),
                           k, B_tags.index(ftag), sgn))
            B_blocks[E].append((DIM[rho], sv, nuE, ntE))
    B_sum = {E: summarize(B_blocks[E]) for E in ES}
    for E in ES:
        print('B E=%d' % E, B_sum[E], flush=True)

    # ---------------- G4
    g4 = {'split_ks': split_ks, 'expected_ks': sorted(EXPECT_SPLIT), 'n_leaves': len(leaves)}
    g4['ks_ok'] = split_ks == sorted(EXPECT_SPLIT)
    g4['n_leaves_ok'] = len(leaves) == 50
    g4['sizes'] = {k: sizes[k] for k in split_ks}
    ok_sz = True
    split_tab = []
    for k in EXPECT_SPLIT:
        n_p, n_m = sizes[k] if len(sizes.get(k, [])) == 2 else (None, None)
        if n_p is None:
            ok_sz = False
            continue
        t_p, t_m = EXPECT_SPLIT[k]
        dp, dm = n_p - t_p, n_m - t_m
        ok = dp >= 0 and dm >= 0 and abs(dp - dm) <= 1 and dp + dm == subs[k].n - (t_p + t_m)
        ok_sz &= bool(ok)
        split_tab.append((k, int(meta[k, 0]), subs[k].n, t_p, t_m, n_p, n_m, dp + dm))
    g4['sizes_ok'] = bool(ok_sz)
    union = []
    uok = True
    for k in split_ks:
        for E in ES:
            par = block_sv(subs[k], E)
            lp, lm = leaf_sv.get(('k%d+' % k, E)), leaf_sv.get(('k%d-' % k, E))
            parts = [x for x in (lp, lm) if x is not None]
            if par is None and not parts:
                continue
            if par is None or not parts:
                # one side empty at this level -> parent has no block, leaves would be zero-size
                union.append(dict(k=k, E=E, ok=False, note='presence mismatch')); uok = False
                continue
            u = np.sort(np.concatenate(parts))[::-1]
            p = par[0]
            n = max(len(u), len(p))
            u = np.concatenate([u, np.zeros(n - len(u))]); p = np.concatenate([p, np.zeros(n - len(p))])
            dev = float(np.max(np.abs(u - p)) / max(p.max(), 1e-30))
            union.append(dict(k=k, E=E, dev=dev, ok=dev <= TOL))
            uok &= dev <= TOL
    g4['union'] = union
    g4['union_ok'] = bool(uok)
    g4['max_union_dev'] = max([u.get('dev', 0) for u in union]) if union else None
    g4['pass'] = bool(g4['ks_ok'] and g4['n_leaves_ok'] and g4['sizes_ok'] and uok)
    print('G4', {k: v for k, v in g4.items() if k != 'union'}, flush=True)

    summary = dict(A=A_sum, B=B_sum,
                   gates=dict(G1=g1, G2=g2, G3=g3, G4=g4), runtime_s=time.time() - t0)
    if not (g1['pass'] and g3['pass'] and g4['pass']):
        fn = os.path.join(OUT, 'Singlet_firstorder_L8_GATEFAIL.json')
        with open(fn, 'w') as f:
            json.dump(summary, f, indent=1, default=str)
        print('GATE FAILURE, wrote', fn, flush=True)
        return

    np.savez_compressed(os.path.join(OUT, 'Singlet_firstorder_L8.npz'), **A_data, **B_data,
                        A_meta=np.array(A_rows, dtype=int), B_meta=np.array(B_rows, dtype=int),
                        B_tags=np.array(B_tags))
    with open(os.path.join(OUT, 'Singlet_firstorder_L8.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)

    def tab(S):
        s = '| E | sub-blocks | distinct | ratios | <r> +- sem | weighted rank |\n|---|---|---|---|---|---|\n'
        for E in ES:
            x = S[E]
            rr = '-' if x['r_mean'] is None else '%.3f +- %.3f' % (x['r_mean'], x['r_sem'])
            s += '| %d | %d | %d | %d | %s | %d |\n' % (E, x['blocks'], x['distinct'], x['ratios'], rr, x['weighted_rank'])
        return s
    md = '# Singlet first-order spectra, Lambda = 8 (th_max = 8)\n\n'
    md += ('Convention: the first-order levels at level E are E +- lam*sigma for the nonzero singular '
           'values sigma, plus (nu_E + nt_E - 2*rank) states at E unshifted. sigma is in the normalization '
           'of the stored M (kappa = sqrt(2)*pi included).\n\n')
    md += '## Resolution A (40 sub-sectors)\n\n' + tab(A_sum) + '\n## Resolution B (fully resolved, hidden Z_2 S, %d leaves)\n\n' % len(leaves) + tab(B_sum)
    md += '\n## S-split sizes (G4)\n\n| k | rho | n parent | table + | table - | leaf + | leaf - | ambiguous states |\n|---|---|---|---|---|---|---|---|\n'
    for r in split_tab:
        md += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    md += ('\nS splits only the Z_R = +1 sub-sectors, which carry the even levels, so B = A at odd E by construction.\n')
    md += '\n## Gates\n\n- G1 %s: %s\n- G2 (info): %s\n- G3 %s: max rel dev %s\n- G4 %s: split ks %s, leaves %d, sizes_ok %s, union max dev %s\n' % (
        'PASS' if g1['pass'] else 'FAIL', json.dumps(g1), json.dumps(g2),
        'PASS' if g3['pass'] else 'FAIL', max([x['rel_dev'] or 0 for x in g3['detail']]),
        'PASS' if g4['pass'] else 'FAIL', split_ks, len(leaves), g4['sizes_ok'], g4['max_union_dev'])
    md += '\nRuntime %.0f s.\n' % (time.time() - t0)
    with open(os.path.join(OUT, 'Singlet_firstorder_L8.md'), 'w') as f:
        f.write(md)
    print('done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
