#!/bin/bash
# run from anywhere: scripts change into src/ and read RUOQU_WORK / RUOQU_PROJECT (see src/paths.py)
cd "$(dirname "$0")/../src" || exit 1
export PROJECT=${RUOQU_PROJECT:-$(cd .. && pwd)}
export WORK=${RUOQU_WORK:-$PROJECT/work}
# Ruoqu-L-disc: direct date-output control with the same size, data, sampling and token budget as l3 (fold 0)
export LETTER_CORPUS=corpus3_f0
name=${1:-disc_l3}; mkdir -p $WORK/runs/$name
[ -f $WORK/runs/$name/model.pt ] && grep -q "^done" $WORK/runs/$name/log.txt 2>/dev/null || \
  { python -u disc.py train --name $name --size l --tokens 1.5e9 --bs 128 --T 256 --lr 4e-4 | tee $WORK/runs/$name/log.txt; echo done >> $WORK/runs/$name/log.txt; }
for s in heldout blind_full blind_res e4_full chapters imit synth; do
  K=24; [ $s = blind_full ] && K=400; [ $s = chapters ] && K=400; [ $s = imit ] && K=3; [ $s = synth ] && K=1
  python -u disc.py score --name $name --split $s --K $K
done
