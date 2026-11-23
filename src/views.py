"""Alternative views of the text, as remapping tables over vocab ids.

skel : function-word skeleton — every character that is not a function word becomes X
lex  : lexical view — every function word becomes X (content characters kept)
Function words are grammatical particles, pronouns, negators, auxiliaries, aspect markers,
classifiers and conjunctions whose use changes over time but is rarely controlled consciously.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import json
import numpy as np

C = (WORK + '/data/') + __import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')
# placeholder X gets the extra id len(vocab); models are built with len(vocab)+1 entries

FUNC = set('之乎者也矣焉哉兮耳而以於于其所為則乃且若或莫弗勿毋不未非否亦皆咸俱悉凡蓋夫豈寧何奚曷胡安惡孰誰盍'
           '與歟邪耶爾云然斯此彼是茲吾我余予朕汝女厥諸旃將已既曾嘗方始遂因故雖縱苟如使令被見把得了著個箇這那'
           '甚麼什便都只祇卻就才纔還又底的地呢嗎罷吧些們每自從向在到去來裏裡頗極最太稍略殊良尤益愈彌更復再並'
           '共同相互惟唯維但特徒直即乍忽俄頃須臾無毋勿休莫休亡罔靡蔑微非匪伊惟維唯式載聿言薄于於乎諸焉旃')


def table(view):
    vocab = json.load(open(f'{C}/vocab.json'))
    V = len(vocab); X_ID = V
    if view == 'full':
        return None
    t = np.arange(V, dtype=np.int64)
    isf = np.array([c in FUNC for c in vocab])
    isf[:3] = False  # pad, sep, unk keep their ids
    if view == 'skel':
        t[(~isf) & (np.arange(V) >= 3)] = X_ID
    elif view == 'lex':
        t[isf] = X_ID
    t[2] = 2
    return t
