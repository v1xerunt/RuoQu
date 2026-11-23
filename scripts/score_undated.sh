#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# date the undated texts with fold models (ensemble in the analysis); texts were never in any training set
# usage: undated_score.sh [run ...]   (default: l3 l3_f1)
runs=${@:-l3 l3_f1}
for run in $runs; do python -u score.py --name $run --split undated --genre unk --K 24; done
