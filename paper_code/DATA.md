# Data

The paper uses two publicly available eye-tracking corpora plus the SUBTLEX-US frequency norms. None are redistributed in this repository — they must be obtained directly from the original sources under their respective licenses.

## Required files

After downloading, place all files under `paper_code/data/` (sibling of `src_v2/`, `archive/`). The expected layout is:

```
paper_code/
├── data/
│   ├── Geco_MonolingualReadingData.csv
│   ├── Geco_EnglishMaterial.csv
│   ├── geco_predictability.pkl
│   ├── Provo_Corpus-Eyetracking_Data.csv
│   ├── Provo_Corpus-Predictability_Norms.csv
│   └── SUBTLEXus.txt
└── ...
```

These paths are referenced from `src_v2/paper_experiments/config.py` (`GECO_*`, `PROVO_FILE`, `SUBTLEX_FILE`). If you place data elsewhere, edit those constants.

## GECO (Ghent Eye-tracking Corpus)

- Reference: Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. *Behavior Research Methods*, 49(2), 602–615.
- Download: <https://expsy.ugent.be/downloads/geco/>
- Files needed:
  - `Geco_MonolingualReadingData.csv` — per-fixation reading data (English monolingual subset; ~150 MB)
  - `Geco_EnglishMaterial.csv` — sentence/word material with positions (~3 MB)
- License: CC-BY (verify with publishers).

### `geco_predictability.pkl`

Per-word cloze predictability values aligned to the GECO material file, used by the linear-regression baseline. This was precomputed for our experiments. If absent, the linear-regression baseline can run with predictability dropped from its feature set; the cascade itself does not consume predictability.

To regenerate, build a dict `{(text_id, sentence_number, word_number): predictability_float}` and pickle it to this path. (Format: `dict` keyed by `(int, int, int)`.)

## Provo

- Reference: Luke, S. G., & Christianson, K. (2018). The Provo corpus: A large eye-tracking corpus with predictability norms. *Behavior Research Methods*, 50(2), 826–833.
- Download: <https://osf.io/sjefs/>
- Files needed:
  - `Provo_Corpus-Eyetracking_Data.csv` (~70 MB)
  - `Provo_Corpus-Predictability_Norms.csv` (~14 MB; only used by some baselines)
- License: per OSF terms.

## SUBTLEX-US

- Reference: Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms. *Behavior Research Methods*, 41, 977–990.
- Download: <https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus> (`SUBTLEXus74286wordstextversion.zip`)
- File needed: `SUBTLEXus.txt` (TSV with `Word` and `FREQcount` columns; ~3 MB).

## Splits used in the paper

- GECO: 30,601 train / 4,494 val / 5,501 test sentences (split by text — see `archive/original_ezreader/geco_loader.py: split_geco`).
- Provo: full corpus held out as cross-corpus generalization test (~2,654 word-aggregated sentences).
- Aggregation: `min_participants=5` per word for sentence-aggregated splits.

These splits are deterministic given the input CSVs and the loader code in this repo; no manual partitioning is needed.

## Symlink convention

Several legacy baseline scripts resolve data via `archive/data/` (a symlink to `data/` at the repo root). The pipeline driver (`pipeline.sh`, phase B preflight) creates this symlink automatically:

```bash
ln -s ../data archive/data
```

If you run individual baselines outside the pipeline, create that symlink yourself.
