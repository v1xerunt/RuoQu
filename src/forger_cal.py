"""False-alarm calibration of the forger model: genuine early held-out texts (fold 0, per-token scores of
early_ids_v3, tag _early) vs the frozen blind list, for every lambda. A false alarm = a genuine text whose
posterior puts > 50% of its mass more than 50 years after its own label upper bound.
Usage: python forger_cal.py <run> [tau]
"""
import sys
import numpy as np
import pandas as pd
import forger as fg
from blind import auc


def main(name, tau=None):
    import evaluate
    if tau is None:
        _, cal = evaluate.evaluate(f'{fg.R}/{name}/scores_heldout_unk.npz')
        tau = float(np.exp(np.mean(np.log(cal['taus']))))
    tau = float(tau)
    pd.set_option('display.width', 200)
    for lam in [1.0 - 1e-9, 0.99, 0.98, 0.95, 0.9]:
        h = fg.heldout(name, lam, 0.2, tau)
        b = fg.blind(name, lam, 0.2, tau)
        a, bb = b[b.cat == 'A'], b[b.cat == 'B']
        fa = (h.p_later > .5).mean()
        # detection of forgeries at the threshold that gives a 5% false-alarm rate on genuine texts
        thr = np.quantile(h.p_later, .95)
        print(f'lam={lam:.3f}: genuine early texts n={len(h)} false alarms {fa:.3f}; A detected {(a.p_later > .5).mean():.2f} '
              f'(at 5% FA threshold {thr:.2f}: {(a.p_later > thr).mean():.2f}); B wrongly flagged {(bb.p_later > .5).mean():.2f}; '
              f'AUC(A vs genuine) {auc(a.p_later, h.p_later):.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
