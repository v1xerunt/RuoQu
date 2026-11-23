#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# fold-0 analyses of the final (v3) model; run after eval_all has scored heldout and blind sets
# usage: scripts/analyses_fold0.sh [run=l3]
run=${1:-l3}
export LETTER_V=3 LETTER_CORPUS=corpus3_f0
B=$WORK/runs/baselines_corpus3_f0
R=$WORK/runs
echo "== E1 (fold 0)"
python -u evaluate.py $run=$R/$run/scores_heldout_unk.npz lr=$B/lr_heldout.npz ngram=$B/ngram_heldout.npz fw=$B/fw_heldout.npz
echo "== blind ($run)"
python -u blind.py $run
for bl in lr ngram fw; do echo "== blind (baseline $bl)"; python -u blind.py bl:$bl | grep -E "^==|AUC"; done
tau=$(python -c "
import numpy as np, evaluate
_, cal = evaluate.evaluate('$R/$run/scores_heldout_unk.npz')
print(float(np.exp(np.mean(np.log(cal['taus'])))))" | tail -n 1)
echo "== forger (tau=$tau)"
python -u forger.py $run $tau
echo "== liezi"
python -u liezi.py $run corpus3_f0 3
echo "== shangshu anatomy"
python -u shangshu_anatomy.py $run corpus3_f0 3
