#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# build the final corpus (LETTER_V=3): variant map, labels, folds, masking, evaluation splits
export LETTER_V=3
python -u varmap.py | tail -2
python -u corpus.py
python -u chapters.py build
python -u imit.py build
LETTER_CORPUS=corpus3_f0 python -u synth.py | tail -2
LETTER_CORPUS=corpus3_f0 python -u editions.py | tail -6
python -c "import pandas as pd; h=pd.read_csv(__import__('os').environ['WORK'] + '/data/corpus3_f0/heldout.tsv',sep='\t'); s=h[(h.lo+h.hi)/2<900]; open(__import__('os').environ['WORK'] + '/runs/early_ids_v3.txt','w').write(','.join(s.id)); print('early ids', len(s))"
