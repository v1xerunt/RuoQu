#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# score imitation / control pieces of the collections held out in this model's fold,
# and genuine pieces of authors near the imitated models' dates (ceiling calibration), if built
python -u score.py --name $1 --split imit --genre unk --K 3 --tokens 4
corpus=$(python -c "import json; print(json.load(open('$WORK/runs/$1/config.json'))['corpus'])")
[ -f $corpus/genuine.bin ] && python -u score.py --name $1 --split genuine --genre unk --K 3
exit 0
