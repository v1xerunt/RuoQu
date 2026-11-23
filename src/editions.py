"""E10: the same held-out work in different base editions (WYG 四庫, SBCK 四部叢刊, tls ...).

Downloads non-master edition branches of fold-0 held-out works, extracts them with the same
cleaning, character normalization and vocabulary, and writes corpus_f0/ed_<EDITION>.{bin,tsv}
plus ed_master.{bin,tsv} (the master text of the same works, unmasked) for a paired comparison.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, subprocess, zipfile, re
import numpy as np
import pandas as pd
from extract import process
from corpus import D

RAW = (WORK + '/raw')
C = f"{D}/{__import__('os').environ.get('LETTER_CORPUS', 'corpus_f0')}"
EDS = ['WYG', 'SBCK', 'tls', 'ZTDZ', 'HFL']


def download(pairs):
    os.makedirs(f'{RAW}/kred', exist_ok=True)
    for i, b in pairs:
        out = f'{RAW}/kred/{i}_{b}.zip'
        if not os.path.exists(out):
            subprocess.run(['curl', '-sL', '-o', out, f'https://codeload.github.com/kanripo/{i}/zip/refs/heads/{b}'], timeout=120)


def main():
    br = {}
    for l in open(f'{RAW}/kred_branches.txt'):
        p = l.split(); br[p[0]] = [b for b in p[1:] if b in EDS]
    held = pd.read_csv(f'{C}/heldout.tsv', sep='\t').set_index('id')
    master_ed = {}
    for part in ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']:
        for l in open(f'{D}/extract/{part}.jsonl'):
            r = json.loads(l)
            if r['id'] in held.index: master_ed[r['id']] = r.get('edition', '')
    pairs = [(i, b) for i, bs in br.items() if i in held.index for b in bs if b != master_ed.get(i)]
    print(len(pairs), 'edition branches to fetch', flush=True)
    if sys.argv[1:2] == ['download']:
        download(pairs); return
    vm = json.load(open(f"{D}/varmap{os.environ.get('LETTER_V', '')}.json"))
    vocab = json.load(open(f'{C}/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    # process: same extract pipeline but on the branch zip
    import extract
    def tok(juans):
        s = 'Ｓ'.join(juans)
        return np.array([1 if c == 'Ｓ' else c2i.get(vm.get(c, c), 2) for c in s], np.uint16)
    out = {e: ([], []) for e in EDS + ['master']}
    for i, b in pairs:
        fn = f'{RAW}/kred/{i}_{b}.zip'
        if not os.path.exists(fn) or os.path.getsize(fn) < 1000: continue
        z = zipfile.ZipFile(fn)
        names = sorted([n for n in z.namelist() if re.search(r'_\d+[a-z]?\.txt$', n)], key=extract.juan_key)
        juans = []
        for n in names:
            raw = z.read(n).decode('utf8', errors='replace')
            mj = re.search(r'#\+PROPERTY: JUAN\s+(.*)', raw)
            if extract.juan_key(n)[0] == 0 or (mj and '提要' in mj.group(1)): continue
            t = extract.clean_juan(raw)
            if t: juans.append(t)
        if not juans: continue
        m = process(i)
        out[b][0].append(i); out[b][1].append(tok(juans))
        out['master'][0].append(f'{i}|{b}'); out['master'][1].append(tok(m['juans']))
    for e, (ids, arrs) in out.items():
        if not ids: continue
        offs = np.cumsum([0] + [len(a) for a in arrs[:-1]])
        np.concatenate(arrs).tofile(f'{C}/ed_{e}.bin')
        base = [x.split('|')[0] for x in ids]
        sub = held.loc[base, ['title', 'lo', 'hi', 'genre']].copy()
        sub.insert(0, 'id', ids); sub['offset'] = offs; sub['length'] = [len(a) for a in arrs]
        sub.to_csv(f'{C}/ed_{e}.tsv', sep='\t', index=False)
        print(e, len(ids), 'texts', sum(len(a) for a in arrs) / 1e6, 'M chars')


if __name__ == '__main__':
    main()
