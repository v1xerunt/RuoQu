#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# per-character log-probabilities over the date grid for a period-stratified sample of held-out texts
# (each text by the fold model that held it out), for the lexical-change timeline figures
R=$WORK/runs
for f in 0 1 2 3 4; do
  run=l3; [ $f -gt 0 ] && run=l3_f$f
  python -u score.py --name $run --split heldout --ids @$R/lex_ids_f$f.txt --K 8 --tokens 4 --tag _lexsample
done
