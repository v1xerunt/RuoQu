"""Save the blind-list work tables (consensus vs model) for the five-fold ensemble, for plotting."""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, numpy as np, pandas as pd
os.environ.setdefault('LETTER_CPREFIX', 'corpus3_f')
import oof
from blind import work_table
import evaluate as ev, blind as bl
cp0 = (WORK + '/data/corpus3_f0'); ev.C = cp0; bl.C = cp0
tab, res, tau = oof.oof('l3')
e = oof.ensemble_scores('l3', 'blind_full')
df = work_table('blind_full', e[0], tau)
df.to_csv((WORK + '/runs/blind_ensemble_table.tsv'), sep='\t', index=False)
print(df[['work', 'cat', 'attr', 'cons', 'median', 'q10', 'q90']].to_string(index=False))
