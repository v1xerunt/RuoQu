"""Data locations.

WORK    holds the large working data: raw texts (raw/), built corpora (data/corpus3_f*/) and model runs (runs/).
        Set RUOQU_WORK; defaults to <repository>/work.
PROJECT holds the curated lists that ship with the repository (data/: blind lists, manual dates, prompt table).
        Set RUOQU_PROJECT; defaults to the repository root.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get('RUOQU_PROJECT', os.path.dirname(_here))
WORK = os.environ.get('RUOQU_WORK', os.path.join(PROJECT, 'work'))
