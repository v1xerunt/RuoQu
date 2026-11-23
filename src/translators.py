"""Buddhist translation register by translator (OOF predictions): do Song translators read like the Tang?
Usage: python translators.py <prefix> (needs runs/<prefix>_oof.tsv and the corpus heldout tables for authors)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, os
import pandas as pd
from freeze import register

R = (WORK + '/runs')
D = (WORK + '/data')
CP = os.environ.get('LETTER_CPREFIX', 'corpus3_f')
# activity of the Song translation bureau (譯經院, 982-1082) and Liao/Song 慈賢; years from the standard catalogues
ACTIVE = {'法天': (973, 1001), '法賢': (982, 1001), '天息災': (980, 986), '施護': (980, 1017), '惟淨': (1009, 1050),
          '慈賢': (1000, 1050), '日稱': (1056, 1078), '法護': (1004, 1058), '紹德': (1060, 1080), '智吉祥': (1070, 1080)}


def main(prefix):
    t = pd.read_csv(f'{R}/{prefix}_oof.tsv', sep='\t')
    au = pd.concat([pd.read_csv(f'{D}/{CP}{f}/heldout.tsv', sep='\t', usecols=['id', 'author', 'genre'])
                    for f in range(5) if os.path.exists(f'{D}/{CP}{f}/heldout.tsv')]).drop_duplicates('id').set_index('id')
    t['author'] = t.id.map(au.author)
    t['register'] = [register(c, g) for c, g in zip(t.cat, t.genre)]
    song = t[(t.lo == 960) & (t.hi == 1279)]
    pd.set_option('display.width', 200)
    print('Texts labelled with the whole Song dynasty (960-1279), OOF median prediction by register:')
    print(song.groupby('register')['median'].agg(['size', 'median']).to_string())
    tr = t[t.author.isin(ACTIVE)].copy()
    tr['act_mid'] = tr.author.map(lambda a: sum(ACTIVE[a]) / 2)
    tr['offset'] = tr['median'] - tr.act_mid
    print('\nSong translators (OOF median prediction; offset from the translator\'s active years):')
    print(tr.groupby('author').agg(n=('median', 'size'), pred=('median', 'median'), offset=('offset', 'median')).to_string())
    print(f'all Song-bureau translations: n={len(tr)}, median prediction {tr["median"].median():.0f}, '
          f'median offset {tr.offset.median():.0f} years')


if __name__ == '__main__':
    main(sys.argv[1])
