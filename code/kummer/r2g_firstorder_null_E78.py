"""r2g_firstorder_null_E78.py -- the matched null of the generic first-order gap ratio at E = 7 and E = 8 with
enough samples to quote.

Same estimator as r2g_firstorder_nullcheck.py (not included in this repository): real Gaussian matrices
of the sub-block shapes (nt x rank, the
Laguerre orthogonal ensemble), all singular values, gap ratios per block, pooled.  Shapes, identical along every
generic direction (qgen, qgen2, qrand106/111/112):
  E = 7  four sub-blocks 7056 x 3536                                   (14136 ratios)
  E = 8  the three quoted sub-blocks 19856 x 9208                      (27618 ratios)
Writes r2_out/Generic_firstorder_null_E78.json after every sample, so a partial run is usable.

Usage: python r2g_firstorder_null_E78.py [n7] [n8]
"""
import json
import os
import sys
import time

import numpy as np

import f2_common as f2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
SHAPES = {7: [(7056, 3536)] * 4, 8: [(19856, 9208)] * 3}


def main(n7=50, n8=20):
    rng = np.random.default_rng(20261103)
    res = {}
    path = os.path.join(OUT, 'Generic_firstorder_null_E78.json')
    for E, nsamp in ((7, n7), (8, n8)):
        samples = []
        for i in range(nsamp):
            t0 = time.time()
            g = np.concatenate([f2.gap_ratios(np.sort(np.linalg.svd(rng.standard_normal(s), compute_uv=False)))
                                for s in SHAPES[E]])
            samples.append(float(g.mean()))
            m, sd = float(np.mean(samples)), float(np.std(samples, ddof=1)) if len(samples) > 1 else float('nan')
            res[str(E)] = dict(shapes=SHAPES[E], n_gaps=int(g.size), nsamp=len(samples), mean=m, sd=sd,
                               se=sd / np.sqrt(len(samples)), samples=samples)
            with open(path, 'w') as fh:
                json.dump(dict(doc=__doc__, seed=20261103, levels=res), fh, indent=1)
            print('E=%d sample %d: %.4f  (running %.4f, sd %.4f)  [%.0f s]' % (E, i, samples[-1], m, sd,
                                                                             time.time() - t0), flush=True)


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:3]))
