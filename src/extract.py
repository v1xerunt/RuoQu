"""Extract main-text Han characters from Kanripo zips.

Removes: org headers, page markers, 四庫 提要 (juan 000), 四庫 考證 appendices, interlinear small notes
"(…/…)" and "(…)", heading lines, punctuation and all non-Han characters.
Output: data/extract/{part}.jsonl  {id, edition, n_juan, juans: [str]}
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, sys, json, zipfile, unicodedata
from multiprocessing import Pool

RAW = (WORK + '/raw')
OUT = (WORK + '/data/extract')

HAN = re.compile(r'[㐀-䶿一-鿿\U00020000-\U0003134f]')
NONHAN = re.compile(r'[^㐀-䶿一-鿿\U00020000-\U0003134f]+')
PB = re.compile(r'<pb:[^>]*>')
NOTE = re.compile(r'\([^()]*\)|（[^（）]*）')  # interlinear notes; applied repeatedly for nesting
# Heading / colophon lines: juan titles with author/editor credits, 四庫 banner lines, etc.
HEAD = re.compile(r'(欽定四庫全書|四庫全書|卷[第之]?[一二三四五六七八九十百〇零上中下\d]+|^目錄$|^目録$)')
CREDIT = re.compile(r'(撰|著|註|注|譯|編|輯|校|纂|述|集|訂|刋|刊|閱|重修|奉敕|奉勑)$')
KAOZHENG = re.compile(r'^.{0,20}卷[第之]?[一二三四五六七八九十百〇\d上中下]*考證$')


def clean_juan(txt):
    lines = txt.split('\n')
    out = []
    for ln in lines:
        if ln.startswith('#') or ln.startswith('*'):
            continue
        ln = PB.sub('', ln).replace('¶', '').replace('|', '')
        ln = unicodedata.normalize('NFKC', ln)
        s = ln.strip().replace(' ', '')
        if not s:
            continue
        core = NONHAN.sub('', NOTE.sub('', s))
        if KAOZHENG.match(core):
            break  # Qing editors' 考證 appended at the end of a juan
        if len(core) <= 30 and (HEAD.search(core) or (CREDIT.search(core) and len(core) <= 16)):
            continue
        out.append(s)
    t = ''.join(out)
    for _ in range(3):
        t2 = NOTE.sub('', t)
        if t2 == t: break
        t = t2
    t = re.sub(r'[()（）]', '', t)
    return NONHAN.sub('', t)


def juan_key(n):
    m = re.search(r'_(\d+)[a-z]?\.txt$', n)
    return (int(m.group(1)) if m else 10**6, n)


def process(zid):
    fn = f'{RAW}/krzip/{zid}.zip'
    try:
        z = zipfile.ZipFile(fn)
    except Exception as e:
        return dict(id=zid, error=str(e))
    names = sorted([n for n in z.namelist() if re.search(r'_\d+[a-z]?\.txt$', n)], key=juan_key)
    juans, edition, skipped = [], '', 0
    for n in names:
        raw = z.read(n).decode('utf8', errors='replace')
        m = re.search(r'#\+PROPERTY: BASEEDITION\s+(\S+)', raw)
        if m and not edition: edition = m.group(1)
        mj = re.search(r'#\+PROPERTY: JUAN\s+(.*)', raw)
        jtag = mj.group(1).strip() if mj else ''
        if juan_key(n)[0] == 0 or '提要' in jtag:
            skipped += 1; continue
        t = clean_juan(raw)
        if t: juans.append(t)
    return dict(id=zid, edition=edition, n_juan=len(juans), skipped=skipped, juans=juans)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(f'{RAW}/krzip') if f.endswith('.zip'))
    parts = sys.argv[1:] or ['KR1', 'KR2', 'KR3', 'KR4', 'KR5', 'KR6']
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    with Pool(nproc) as pool:
        for part in parts:
            sub = [i for i in ids if i.startswith(part)]
            n_chars = 0
            with open(f'{OUT}/{part}.jsonl', 'w') as f:
                for r in pool.imap(process, sub, chunksize=4):
                    n_chars += sum(len(j) for j in r.get('juans', []))
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(part, len(sub), 'texts', n_chars / 1e6, 'M chars', flush=True)
