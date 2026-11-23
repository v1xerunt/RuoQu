#!/bin/bash
WORK=${RUOQU_WORK:-$(cd .. && pwd)/work}
# score a trained model on every evaluation split (run inside a GPU job)
name=$1
corpus=$(python -c "import json;print(json.load(open('$WORK/runs/$name/config.json'))['corpus'])")
if [ -f $corpus/chapters.bin ]; then python -u score.py --name $name --split chapters --genre unk --K 400 --tokens 4; fi
python -u score.py --name $name --split heldout --genre unk
python -u score.py --name $name --split blind_full --genre unk --K 400 --tokens 4
python -u score.py --name $name --split blind_full --genre true --K 400
python -u score.py --name $name --split blind_res --genre unk --K 400
python -u score.py --name $name --split e4_full --genre unk --K 400 --tokens 4
python -u score.py --name $name --split e4_res --genre unk --K 400
if [ -f $corpus/synth.bin ]; then python -u score.py --name $name --split synth --genre unk --K 1 --tokens 1; fi
for e in $corpus/ed_*.bin; do [ -f "$e" ] && python -u score.py --name $name --split $(basename $e .bin) --genre unk --K 24; done
