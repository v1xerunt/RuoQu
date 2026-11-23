"""Scholarly consensus dates vs model dates, for the scatter plot (written by hand from the literature checks in
data/audit_cases_v1.tsv; every row cites its source). basis: 'text' = the date scholars give for the work as
transmitted (its main layer); 'layer' = the date of a later layer that scholars identify in the transmitted text.
status: consensus / disputed (scholars disagree; the row gives one side, named in the source column).
Also adds the frozen blind list (blind_v0: consensus dates for forged (A), rehabilitated (B), open (D) works)
with the five-fold ensemble dates (runs/blind_ensemble_table.tsv).
Output: data/consensus_dates_v0.tsv
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import pandas as pd
import numpy as np

R = (WORK + '/runs')
P = (PROJECT + '/data')
# id, title, sch_lo, sch_hi, basis, status, source
ROWS = [
    ('KR3e0014', '銀海精微', 1400, 1500, 'text', 'consensus', 'Unschuld & Kovacs 1999（15世紀）；四庫：宋以後書'),
    ('KR5a0011', '高上玉皇本行集經（張良校正本）', 1217, 1225, 'text', 'consensus', '謝聰輝 2009；Schachter 2015'),
    ('KR6p0097', '東海若解', 1686, 1734, 'text', 'consensus', 'CBETA X1178：清實賢解'),
    ('KR6b0057', '撰集百緣經', 445, 594, 'text', 'consensus', '出本充代 1995/98；辛嶋 2006；Nattier 2008'),
    ('KR6b0002', '菩薩本緣經', 317, 420, 'text', 'consensus', '顏洽茂、熊娟 2010（西晉以後，或東晉）'),
    ('KR6j0761', '瑜伽集要燄口施食儀', 1150, 1368, 'text', 'consensus', '徐東軍 2020（西夏不動金剛重集）；陸韋志 2022（元代本）'),
    ('KR3a0039', '上蔡語錄', 1159, 1168, 'layer', 'consensus', '朱熹後序1159、後記1168（今本為朱熹編定本）'),
    ('KR6q0053', '臨濟錄', 1100, 1120, 'layer', 'consensus', '柳田聖山：今本源於1120年宗演重刊本'),
    ('KR5c0319', '鉛汞甲庚至寶集成', 1115, 1368, 'text', 'consensus', '陳國符：金代或更晚；孟乃昌：宋後'),
    ('KR3g0025', '靈城精義', 1368, 1550, 'text', 'consensus', '四庫：其非明以前書確矣；劉基注亦明中葉以後偽書'),
    ('KR2k0091', '吳地記', 1100, 1300, 'layer', 'consensus', '四庫：此書不出廣微；今本含淳熙1186、元貞1295等宋元增補'),
    ('KR3f0038', '夏侯陽算經', 762, 780, 'text', 'consensus', '錢寶琮 1963：今本為唐韓延書'),
    ('KR4d0065', '周元公集', 1189, 1274, 'layer', 'consensus', '粟品孝；姜龍翔 2020：宋人編集（諸儒議論等後人附益）'),
    ('KR1h0008', '論語筆解', 1023, 1063, 'layer', 'consensus', '金培懿 2010：北宋許勃繕校重編'),
    ('KR1a0002', '子夏易傳', 880, 1077, 'text', 'consensus', '陳鴻森 1985：十一卷本成於神宗朝以前；晁說之：唐張弧'),
    ('KR6r0006', '十六大羅漢因果識見頌', 1000, 1048, 'text', 'consensus', '范仲淹1048序；印順：中土雜湊'),
    ('KR6i0559', '圓覺經略疏鈔', 1041, 1146, 'layer', 'consensus', '思齊1041校刊序；觀復1146：刊板添改'),
    ('KR6k0013', '曇無德律部雜羯磨 T1432', 410, 500, 'text', 'consensus', '平川彰（四分律節本，5世紀）；Nattier 2008'),
    ('KR6k0014', '羯磨 T1433', 400, 500, 'text', 'consensus', '平川彰；Fang & Lu 2023'),
    ('KR6i0250', '禪祕要法經 T613', 424, 442, 'text', 'consensus', '僧祐錄：曇摩蜜多；境野1935；Schuster 1984'),
    ('KR6k0076', '梵網經 T1484', 440, 480, 'text', 'consensus', 'Benn 1998；大野1954；船山徹'),
    ('KR6j0118', '首楞嚴經 T945', 713, 720, 'text', 'consensus', 'Keyworth 2022（713年後）；Benn 2008'),
    ('KR6q0113', '達磨大師血脉論', 780, 850, 'text', 'consensus', '中華佛教百科（寶林傳801前後）；柳田（牛頭宗）'),
    ('KR6q0115', '達磨大師破相論（觀心論）', 675, 810, 'text', 'consensus', '慧琳一切經音義（約810）著錄神秀作；楊曾文'),
    ('KR3g0020', '葬書', 960, 1279, 'text', 'consensus', '四庫：自宋始出；楊玉成（龍脈觀念中晚唐後）'),
    ('KR3g0024', '天玉經', 960, 1279, 'text', 'consensus', '四庫：至宋始出'),
    ('KR3g0033', '李虛中命書', 960, 1279, 'layer', 'consensus', '四庫：後半言四柱，起於宋，宋人闌入'),
    ('KR3e0033', '太平惠民和劑局方', 1107, 1252, 'text', 'consensus', '四庫：大觀核心＋紹興、寶慶、淳祐增方'),
    ('KR6m0038', '肇論', 550, 589, 'layer', 'consensus', '塚本善隆 1954（梁末至陳合編）'),
    ('KR6i0294', '菩薩瓔珞經 T656', 376, 417, 'text', 'consensus', '僧祐錄竺佛念；林謙、何書群（Radich）2020：竺佛念自撰'),
    ('KR6k0105', '菩薩瓔珞本業經 T1485', 480, 500, 'text', 'consensus', '船山徹 2004；Buswell 1990；大野 1954（中土撰述）'),
    ('KR6f0017', '佛說決定毘尼經 T325', 400, 433, 'text', 'consensus', '大野 1954（東晉後期）；Williams 2002（曇無讖）'),
    ('KR6i0551', '大方廣圓覺修多羅了義經 T842', 700, 730, 'text', 'consensus', 'Muller（8世紀初中土撰述）；Buswell 1990'),
    ('KR2e0017', '汝南遺事', 1234, 1244, 'text', 'consensus', '四庫；Chan 1974（模型偏晚，疑乾隆改譯）'),
    ('KR3l0104', '集異記', 817, 837, 'text', 'consensus', '四庫：蓋猶舊本'),
    ('KR3i0047', '蟹略', 1184, 1231, 'text', 'consensus', '直齋書錄解題；四庫（模型誤差）'),
    ('KR1a0090', '周易文詮', 1339, 1369, 'text', 'consensus', '四庫（未能解釋的偏差）'),
    ('KR6i0526', '佛說演道俗業經 T820', 222, 253, 'text', 'disputed', '齋藤 2000/01：極可能支謙譯（傳統歸聖堅）'),
    ('KR6i0438', '佛說除恐災患經 T744', 254, 260, 'text', 'disputed', '水野（大野–丸山）：或即白延譯本（傳統歸聖堅）'),
    ('KR6i0172', '佛說佛大僧大經 T541', 220, 316, 'text', 'consensus', '林屋 1941：魏吳—西晉失譯（傳統歸沮渠京聲）'),
    ('KR6j0206', '阿難陀目佉尼呵離陀經 T1013', 265, 380, 'text', 'consensus', '林屋 1941/45；稻垣 1987：西晉或更早'),
    ('KR6b0050', '佛本行經 T193', 260, 402, 'text', 'consensus', '干潟（支謙與羅什之間）；後藤 2007'),
    ('KR2b0006', '大唐創業起居注', 705, 710, 'layer', 'disputed', '朱振宏 2020：中宗朝改訂（仇鹿鳴異議；原書約622）'),
    ('KR3l0003', '朝野僉載', 978, 1000, 'text', 'consensus', '今本六卷全從太平廣記輯出（趙庶洋 2023）；四庫：宋人摘錄、已非原書'),
    ('KR3e0010', '肘後備急方', 500, 1144, 'text', 'consensus', '葛洪原書＋陶弘景補闕（約500）＋唐人改動＋楊用道附廣（1144）（Bradley 2022；四庫）'),
    ('KR3i0045', '禽經', 1042, 1102, 'text', 'consensus', '四庫：宋人（王安石學派）偽作，陸佃（1042–1102）首引；張華注亦偽'),
    ('KR1j0008', '廣雅', 581, 900, 'text', 'consensus', '張揖原文附曹憲（隋）音釋以存，含開成後反切的傳刻改動（四庫）'),
    ('KR2k0008', '明一統志', 1522, 1588, 'text', 'consensus', '四庫：多及嘉靖、隆慶建置，後人續入；巴兆祥：萬曆萬壽堂本增至1588'),
    ('KR3e0031', '小兒衞生總微論方', 1150, 1216, 'text', 'consensus', '宋闕名；何大任嘉定序（約1216？）；明1489朱臣刻本'),
    ('KR3e0034', '傳信適用方', 1180, 1180, 'text', 'consensus', '吳彥夔，淳熙庚子1180（直齋）；附夏子益奇疾方38道'),
    ('KR5b0145', '上清骨髓靈文鬼律', 1100, 1116, 'text', 'consensus', 'S&V（Andersen）：饒洞天為傳說祖師，今本為鄧有功（1116前）校編；李志鴻 2009'),
    ('KR5g0036', '太上助國救民總真祕要', 1116, 1116, 'text', 'consensus', 'S&V：元妙宗編，序1116，未見後人層'),
    ('KR5b0248', '太上慈悲道場滅罪水懺', 618, 1279, 'text', 'disputed', 'S&V（Lagerwey）：唐；任繼愈：約唐宋間'),
    ('KR5a0049', '洞真太極北帝紫微神咒妙經', 317, 420, 'text', 'disputed', 'S&V（Schipper）：東晉（文中稱「大晉」）；任繼愈：約唐代'),
    ('KR5c0098', '道德真經注疏', 1100, 1200, 'text', 'consensus', 'S&V（Robinet）：二十三家注的選集，最早12世紀，題顧歡為明藏編者之誤；主體為河上公注與成玄英疏'),
    ('KR3a0030', '二程遺書', 1168, 1168, 'text', 'consensus', '朱熹次錄，乾道四年1168（四庫）'),
    ('KR3a0031', '二程外書', 1173, 1173, 'text', 'consensus', '朱熹編，乾道九年1173（四庫）'),
]


def main():
    a = pd.read_csv(f'{R}/l3_audit.tsv', sep='\t').set_index('id')
    rows = []
    for i, t, lo, hi, basis, status, src in ROWS:
        r = a.loc[i] if i in a.index else None
        rows.append(dict(id=i, title=t, set='audit', cat_lo=r.lo if r is not None else np.nan, cat_hi=r.hi if r is not None else np.nan,
                         sch_lo=lo, sch_hi=hi, basis=basis, status=status, source=src,
                         model_median=r['median'] if r is not None else np.nan, model_q10=r.q10 if r is not None else np.nan,
                         model_q90=r.q90 if r is not None else np.nan, model='Ruoqu-L 5-fold OOF'))
    try:
        b = pd.read_csv(f'{R}/blind_ensemble_table.tsv', sep='\t')
        for r in b.itertuples():
            clo, chi = [float(x) for x in str(r.cons).split('~')]
            alo, ahi = [float(x) for x in str(r.attr).split('~')]
            rows.append(dict(id=r.work, title=r.work, set=f'blind_{r.cat}', cat_lo=alo, cat_hi=ahi, sch_lo=clo, sch_hi=chi, basis='text',
                             status='consensus' if r.cat in ('A', 'B', 'C') else 'open', source='data/blind_v0.tsv（打分前凍結）',
                             model_median=r.median, model_q10=r.q10, model_q90=r.q90, model='Ruoqu-L 5-fold ensemble'))
    except FileNotFoundError:
        print('blind table not ready')
    df = pd.DataFrame(rows)
    df = df[df.status == 'consensus'].reset_index(drop=True)  # the scatter keeps only cases with a scholarly consensus
    df['sch_mid'] = (df.sch_lo + df.sch_hi) / 2
    df['err'] = df.model_median - df.sch_mid
    df['in_interval'] = (df.model_median >= df.sch_lo - 50) & (df.model_median <= df.sch_hi + 50)
    df.to_csv(f'{P}/consensus_dates_v0.tsv', sep='\t', index=False)
    pd.set_option('display.width', 200)
    for s, g in df[df.status != 'open'].groupby('set'):
        print(f'{s}: n={len(g)}, median |model - scholarly midpoint| {g.err.abs().median():.0f} y, within ±50 y of the scholarly range {g.in_interval.mean():.2f}, '
              f'Spearman {g[["model_median", "sch_mid"]].corr("spearman").iloc[0, 1]:.2f}')
    print(df[df.set == 'audit'][['title', 'cat_lo', 'cat_hi', 'sch_lo', 'sch_hi', 'model_median', 'err']].to_string(index=False))


if __name__ == '__main__':
    main()
