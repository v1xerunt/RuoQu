#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# all human-imitation analyses on the rebuilt v3 imitation split (and the AI comparison)
export LETTER_V=3
echo "== imit_eval"; python -u imit_eval.py l3 2>&1 | grep -v Warning
echo "== imit_robust"; python -u imit_robust.py 2>&1 | grep -v Warning
echo "== char_betray"; python -u char_betray.py 2>&1 | grep -v Warning
