#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
python -u score.py --name $1 --split heldout --genre unk --K 24 --tokens 4 --ids @$WORK/runs/early_ids_v3.txt --tag _early
