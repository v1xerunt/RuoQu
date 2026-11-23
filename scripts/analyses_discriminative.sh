#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# Ruoqu-L vs the direct date-output control (fold 0, same held-out set)
export LETTER_V=3 LETTER_CORPUS=corpus3_f0
R=$WORK/runs
echo "== E1"; python -u evaluate.py Ruoqu-L=$R/l3/scores_heldout_unk.npz disc=$R/disc_l3/scores_heldout_unk.npz
echo "== blind (disc)"; python -u blind.py disc_l3 | grep -E "^==|AUC|guwen|jinwen|Shangshu"
