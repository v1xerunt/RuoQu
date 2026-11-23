#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# freezing: grammar (skeleton view: content characters masked) vs lexicon (lexical view: function words masked)
export LETTER_V=3 LETTER_CPREFIX=corpus3_f
for v in skel_m3 lex_m3; do echo "== OOF $v"; python -u oof.py $v > $WORK/runs/oof_$v.txt; sed -n 1,12p $WORK/runs/oof_$v.txt; echo "== rate $v"; python -u freeze.py $v corpus3_f; done
python -u - <<'PY'
import pandas as pd, numpy as np
R = __import__('os').environ['WORK'] + '/runs'
s = pd.read_csv(f'{R}/skel_m3_oof.tsv', sep='\t').set_index('id'); l = pd.read_csv(f'{R}/lex_m3_oof.tsv', sep='\t').set_index('id')
d = s[['title', 'cat', 'genre', 'lo', 'hi', 'median']].join(l[['median']], rsuffix='_lex', how='inner')
d['mid'] = (d.lo + d.hi) / 2; d = d[(d.hi - d.lo) <= 100]
d['e_skel'] = (d['median'] - d.mid).abs(); d['e_lex'] = (d['median_lex'] - d.mid).abs()
d['period'] = pd.cut(d.mid, [-600, -221, 25, 220, 589, 907, 1127, 1279, 1368, 1644, 1950],
                     labels=['戰國', '秦西漢', '東漢', '魏晉南北朝', '隋唐', '五代北宋', '南宋', '元', '明', '清'])
g = d.groupby('period', observed=True).agg(n=('e_skel', 'size'), skel=('e_skel', 'median'), lex=('e_lex', 'median'))
g['ratio_skel_over_lex'] = g.skel / g.lex
pd.set_option('display.width', 200)
print('== datability by period (median absolute error, texts with labels <= 100 years, both views out-of-fold)'); print(g.round(2).to_string())
d.to_csv(f'{R}/views_skel_lex.tsv', sep='\t')
PY
