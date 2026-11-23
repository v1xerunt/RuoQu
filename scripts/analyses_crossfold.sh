#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# cross-fold analyses of the final (v3) models with whatever folds are finished
export LETTER_V=3 LETTER_CPREFIX=corpus3_f
echo "== OOF (E1 cross-fitted) + blind ensembles"; python -u oof.py l3
echo "== audit"; python -u audit.py l3
echo "== freeze offsets"; python -u -c "
import pandas as pd, freeze; pd.set_option('display.width', 250)
freeze.offsets(__import__('os').environ['WORK'] + '/runs/l3_oof.tsv')"
echo "== canon"; python -u canon.py l3 | tail -n 12
