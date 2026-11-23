# Ruoqu (若璩)

Code for

> Gao, J., Zhu, Y., Yu, L., Wang, Y. & Ma, L. *A generative model dates Chinese texts and reveals the limits of imitation.*

Ruoqu is a character-level Transformer decoder conditioned on the year of composition. It learns how Chinese was
written in every century from 600 BCE to 1900 CE and dates a text by Bayes' rule, asking under which years the text
becomes most probable. Because it assigns a probability to every character, the same model gives character-level
evidence for its dates and a model of imitation in which archaism is cheap and anachronism is costly. Models are
trained from scratch on primary texts only, quotations of earlier texts are removed from the training data, and five
models are cross-fitted by author so that every work is dated by a model that never saw it.

The model is named after Yan Ruoqu 閻若璩 (1636–1704), who showed that the Old Text chapters of the *Book of
Documents* were a fourth-century forgery.

- **Results archive** (curated lists, out-of-sample dates, audit flags, imitation pieces, dates for undated texts,
  figure source data): https://doi.org/10.5281/zenodo.24451962
- **Texts**: Kanseki Repository (https://www.kanripo.org) and CBETA (https://www.cbeta.org)

## Repository layout

```
src/            model, training, dating and all analyses (one script per step)
scripts/        shell pipelines that chain the steps in the order used for the paper
data/           curated lists read by the code (blind-test lists, manual date fixes, LLM prompt table, audit cases)
paper_figures/  scripts that draw every figure and table of the paper from the results archive
```

| Step | Scripts |
|---|---|
| Corpus | `catalog.py` and `meta.py` (catalogue dates), `extract.py` (texts), `varmap.py` and `edmarkers.py` (character normalization), `corpus.py` (labels, folds by author, quotation and protection masking), `chapters.py`, `editions.py`, `imit.py`, `aiimit.py`, `undated.py` (evaluation splits) |
| Model | `model.py` (time-conditioned decoder), `views.py` (grammatical and lexical views), `train.py`, `score.py` (likelihood on the year grid), `disc.py` (Aeneas-style discriminative control) |
| Dating and controls | `evaluate.py`, `oof.py` (cross-fitted dates, fold ensembles), `baselines.py`, `blind.py`, `blind_table.py`, `forger.py` and `forger_cal.py` (model of imitation), `synth.py` and `synth_eval.py` (planted anachronisms), `edcmp.py` (editions), `anachron.py` (character-level evidence) |
| Philological tests | `shangshu_anatomy.py`, `liezi.py` |
| Audit | `audit.py`, `audit_evidence.py`, `firstattest.py`, `cbc_fetch.py` and `cbc_attr.py` (CBC@ attributions), `siku_attr.py`, `consensus_table.py` |
| Imitation | `imit_eval.py`, `imit_robust.py`, `aiimit_gen.py` (open-weight generation), `aiimit_eval.py`, `char_betray.py` |
| Undated texts | `byline.py`, `cbeta_fetch.py`, `dila_persons.py`, `daoinfo_fetch.py`, `daoinfo_parse.py`, `undated_eval.py`, `undated_overlap.py`, `new_dates.py` |
| Written language | `freeze.py`, `translators.py`, `canon.py`, `lexical_timeline.py` |

## Installation

```bash
pip install -r requirements.txt
```

Training used PyTorch with bfloat16 on four NVIDIA A100 GPUs per model. `aiimit_gen.py` additionally needs `vllm`.

## Data locations

Two environment variables control where data are read and written (see `src/paths.py`):

- `RUOQU_WORK` (default `./work`) holds the large working data: raw texts in `raw/` (the Kanripo repositories and
  catalogue, CBETA metadata, CBC@ pages), built corpora in `data/corpus3_f0` … `data/corpus3_f4`, and model runs in
  `runs/<name>/`.
- `RUOQU_PROJECT` (default: the repository root) holds the curated lists in `data/`. To rerun the language-model
  imitation analyses, copy `data/aiimit/` from the results archive into this folder.

Further variables select the corpus version and fold: `LETTER_V=3` (final character normalization),
`LETTER_CORPUS=corpus3_f0` (one fold) and `LETTER_CPREFIX=corpus3_f` (all folds).

## Reproducing the paper

```bash
export RUOQU_WORK=/path/to/work LETTER_V=3

# 1. corpus: catalogue dates, texts, normalization, five folds, masking, evaluation splits
cd src && python catalog.py && python meta.py && python extract.py && cd ..
scripts/build_corpus.sh

# 2. train the five cross-fitted Ruoqu-L models (each run scores all evaluation splits when it finishes)
cd src
for k in 0 1 2 3 4; do
  name=l3; [ $k -gt 0 ] && name=l3_f$k
  LETTER_CORPUS=corpus3_f$k torchrun --nproc_per_node 4 train.py --size l --name $name --tokens 1.5e9
done
cd ..
#    (Ruoqu-M: --size m; Ruoqu-XL: --size xl; grammatical and lexical views: --view skel / --view lex)

# 3. analyses
scripts/analyses_fold0.sh l3          # baselines, blind test, model of imitation, Liezi, Shangshu
scripts/analyses_crossfold.sh         # cross-fitted dates, audit, register offsets, canonization
scripts/train_discriminative.sh       # Aeneas-style control
scripts/analyses_discriminative.sh
scripts/analyses_controls.sh          # false-alarm calibration, planted anachronisms, editions
scripts/eval_imit.sh l3               # human imitations (run for each fold model)
scripts/score_llm_imitations.sh       # language-model imitations
scripts/analyses_imitation.sh
scripts/analyses_views.sh             # grammar versus vocabulary
scripts/analyses_masking.sh           # 12- versus 8-character quotation masking
scripts/score_undated.sh l3 l3_f1 l3_f2 l3_f3 l3_f4
```

The run named `l3` in the analysis scripts is the fold-0 model, and `l3_f1` … `l3_f4` are the other folds.
Per-character scores for the audit, the word profiles and the calibration texts come from
`scripts/score_audit_tokens.sh`, `scripts/score_lexical_sample.sh` and `scripts/eval_tok_early.sh`.

## Figures

```bash
export RUOQU_RESULTS=/path/to/RuoQu_results   # the unpacked results archive
cd paper_figures && python fig1.py && python fig2.py && python fig3.py && python fig4.py && python fig5.py && python edfigs.py
```

Figures are written to `paper_figures/output/`.

## Licence

Code is released under the MIT licence. The results archive is released under CC BY 4.0.
