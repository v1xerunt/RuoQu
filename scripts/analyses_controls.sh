#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
export LETTER_V=3 LETTER_CORPUS=corpus3_f0
echo "== forger calibration (full)"; python -u forger_cal.py l3
echo "== forger calibration (lex)"; python -u forger_cal.py lex_l3
echo "== synth (planted anachronisms)"; python -u synth_eval.py l3
echo "== synth (lex)"; python -u synth_eval.py lex_l3
echo "== editions"; python -u edcmp.py l3
