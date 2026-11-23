#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# AI imitations: normalize all generated pieces, then score them with every finished v3 fold model
# (the pieces are in no training set, so any fold model can score them; an ensemble is formed in the analysis)
python -u aiimit.py prepare || exit 1
for run in l3 l3_f1 l3_f2 l3_f3 l3_f4; do
  [ -f $WORK/runs/$run/scores_heldout_unk.npz ] || continue  # written only after training ends
  python -u score.py --name $run --split aiimit --K 3 --tokens 4
done
