"""r2g_wall_scan_targets.py -- 24 directions placed at chosen angles from a wall, to map the crossover near the walls
where the unbiased sample of r2g_wall_scan.py is sparse (the fraction of S^3 within delta of a wall grows only like
0.36 delta/deg, so a log axis has ten times fewer random directions per decade to the left).

Angles: 24 values log-spaced from 0.003 to 3 degrees; even entries near a coordinate wall q_mu = 0, odd entries near a
diagonal wall q_mu = +-q_nu (so each kind spans the range).  For each: a wall of that kind at random, a point q_w drawn
uniformly on it (a normalised Gaussian vector with its normal component removed), q = cos(delta) q_w + s sin(delta) n
with a random sign s.  Accepted only if that wall is the nearest one and every other wall is at least 3 degrees away,
so that a single wall acts (at E = 5 the crossover is halfway at about 0.5 degrees).  These directions are NOT a random
sample: they are drawn and marked separately and enter no statistic of the sample.

Output: r2_out/Wall_scan_targets.json  (input of r2g_wall_scan.py --qfile); with --n=<N> other than 24 the
        same recipe with N angles, written to r2_out/Wall_scan_targets<N>.json (the 20-direction set: --n=20).
        --swap=<name>:<delta> (with --n) replaces the entry <name> of that set by a new direction of the same kind of
        wall at the angle <delta> (degrees), drawn by the same recipe from a separate stream, and writes the next set:
        r2_out/Wall_scan_targets<N>b.json from Wall_scan_targets<N>.json with default_rng([SEED, 1]); with --base=b,
        Wall_scan_targets<N>c.json from Wall_scan_targets<N>b.json with default_rng([SEED, 2]); and so on.  (Above
        3 degrees the condition on the other walls holds as soon as the chosen wall is the nearest.)  Used twice for the
        Letter's Fig. 2:
        --n=20 --swap=near03_diag:0.001 (the open square at 0.0089
        deg replaced by one at 0.001 deg -> 20b), then --n=20 --base=b --swap=near06_coord:3.8 (the filled square at
        0.0266 deg replaced by one at 3.8 deg -> 20c; tried before at 4 and 3.5 deg, the same point of the wall,
        <r> 0.543 and 0.529; --swap=near06_coord:4 or :3.5 reproduces those).
Usage:  python r2g_wall_scan_targets.py [--n=20] [--base=b] [--swap=near06_coord:3.8]
"""
import json
import os
import sys

import numpy as np

from r2g_wall_scan import WALLS, wall_angles

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
SEED, N, DMIN, DMAX, SECOND = 20261105, 24, 0.003, 3.0, 3.0
NORMALS = np.array([w[1] for w in WALLS])
POOLS = {'coord': [k for k, w in enumerate(WALLS) if w[0].endswith('=0')],
         'diag': [k for k, w in enumerate(WALLS) if not w[0].endswith('=0')]}


def draw(rng, dl, kind, name):
    """One direction at the angle dl (degrees) from a random wall of the given kind; the acceptance loop of the
    recipe (that wall nearest, every other wall at least SECOND degrees away)."""
    tries = 0
    while True:
        tries += 1
        k = int(rng.choice(POOLS[kind]))
        n = NORMALS[k]
        g = rng.standard_normal(4)
        qw = g - (n @ g) * n
        qw /= np.linalg.norm(qw)
        s = 1.0 if rng.random() < 0.5 else -1.0
        q = np.cos(np.radians(dl)) * qw + s * np.sin(np.radians(dl)) * n
        ang = wall_angles(q)
        o = np.argsort(ang)
        if o[0] == k and ang[o[1]] >= SECOND:
            break
    assert abs(ang[k] - dl) < 1e-9 * max(dl, 1.0)
    d = dict(name=name, q=q.tolist(), kind=kind, wall=WALLS[k][0], delta=float(dl), second=float(ang[o[1]]),
             tries=tries)
    print('%-12s wall %-7s delta %8.4f deg  second wall %5.2f deg  (%d tries)' % (name, WALLS[k][0], dl, ang[o[1]], tries))
    return d


def main(ndir=N):
    rng = np.random.default_rng(SEED)
    deltas = DMIN * (DMAX / DMIN) ** (np.arange(ndir) / (ndir - 1))
    out = []
    for i, dl in enumerate(deltas):
        kind = 'coord' if i % 2 == 0 else 'diag'
        out.append(draw(rng, dl, kind, 'near%02d_%s' % (i, kind)))
    with open(os.path.join(OUT, 'Wall_scan_targets%s.json' % ('' if ndir == N else str(ndir))), 'w') as fh:
        json.dump(dict(doc=__doc__.split('\n\n')[0], seed=SEED, n=ndir, delta_range=[DMIN, DMAX],
                       min_second_wall=SECOND, directions=out), fh, indent=1)


def swap(ndir, name, dl, base=''):
    """Replace the entry <name> of Wall_scan_targets<ndir><base>.json by a direction of the same kind of wall at the
    angle dl (degrees) and write the next set (the letter after <base>: b, c, ...), the new direction drawn from
    default_rng([SEED, k]) with k = 1 for b, 2 for c, ...; recorded under 'swapped' (b) or 'swapped_<letter>'."""
    tag = '' if ndir == N else str(ndir)
    nxt = chr(ord(base) + 1) if base else 'b'
    k = ord(nxt) - ord('a')
    d = json.load(open(os.path.join(OUT, 'Wall_scan_targets%s%s.json' % (tag, base))))
    i = [x['name'] for x in d['directions']].index(name)
    old = d['directions'][i]
    new = draw(np.random.default_rng([SEED, k]), dl, old['kind'], '%s%s_%s' % (name.split('_')[0], nxt, old['kind']))
    d['directions'][i] = new
    d['swapped' if nxt == 'b' else 'swapped_' + nxt] = dict(
        removed=old, added=new['name'], stream='default_rng([%d, %d])' % (SEED, k),
        note='6 Nov 2026, letter Fig. 2: %s (%.4f deg) replaced by %s (%.4f deg)'
             % (old['name'], old['delta'], new['name'], new['delta']))
    with open(os.path.join(OUT, 'Wall_scan_targets%s%s.json' % (tag, nxt)), 'w') as fh:
        json.dump(d, fh, indent=1)


if __name__ == '__main__':
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    nd = int(opts.get('n', N))
    if 'swap' in opts:
        nm, dl = opts['swap'].split(':')
        swap(nd, nm, float(dl), opts.get('base', ''))
    else:
        main(nd)
