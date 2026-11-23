#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# causal test of the canonization effect: identical medium models trained with 12- vs 8-character quotation masking
export LETTER_V=3 LETTER_CORPUS=corpus3_f0
R=$WORK/runs
echo "== E1"; python -u evaluate.py q12=$R/m3_q12/scores_heldout_unk.npz q8=$R/m3_q8/scores_heldout_unk.npz | grep -E "model|all|先秦|秦漢"
for m in m3_q12 m3_q8; do
  echo "== canon $m"; python -u canon.py $m | sed -n '/禮記: per-character/,$p'
done
