# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Binary classification of tweets as disaster-related (1) or not (0), using data from the [Kaggle NLP Getting Started competition](https://www.kaggle.com/c/nlp-getting-started).

Two models are implemented and compared:
- **Logistic Regression** — TF-IDF (5000 features, unigrams + bigrams) → scikit-learn LogisticRegression. ~82% val accuracy.
- **DistilBERT** — Fine-tuned `distilbert-base-uncased` on cleaned text + keyword + location. ~80.7% val accuracy over 3 epochs.

---

## Directory Layout

```
├── data/
│   ├── raw/          # Original CSVs (committed)
│   └── processed/    # gitignored — written at runtime
├── src/
│   ├── preprocessing.py   # Shared text cleaning pipeline
│   ├── features.py        # TF-IDF vectorizer helpers
│   ├── evaluate.py        # Metrics + plot functions
│   ├── utils.py           # Seed, logger, path constants
│   ├── train.py           # CLI training entry point
│   ├── predict.py         # CLI inference entry point
│   └── models/
│       ├── lr_model.py    # LogisticRegressionModel class
│       └── bert_model.py  # DistilBertClassifier class
├── config/
│   ├── lr_config.yaml     # LR hyperparameters and paths
│   └── bert_config.yaml   # BERT hyperparameters and paths
├── notebooks/
│   ├── 01_eda.ipynb           # EDA only — no training
│   ├── 02_experiments.ipynb   # Thin wrapper calling src/
│   ├── Logistic_Regression model.ipynb   # Original notebook (reference)
│   └── BERT model.ipynb                  # Original notebook (reference)
├── models/           # gitignored — saved model artifacts
├── outputs/
│   ├── figures/      # gitignored — saved plots
│   └── submissions/  # gitignored — submission CSVs
└── requirements.txt
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train a model
```bash
# Logistic Regression
python -m src.train --model lr --config config/lr_config.yaml

# DistilBERT (GPU recommended)
python -m src.train --model bert --config config/bert_config.yaml
```

### Generate test predictions
```bash
python -m src.predict --model lr   --config config/lr_config.yaml
python -m src.predict --model bert --config config/bert_config.yaml
```

Submission files land in `outputs/submissions/`.

### Notebooks
```bash
cd notebooks
jupyter notebook
```
Run `01_eda.ipynb` for data exploration, `02_experiments.ipynb` to train interactively and compare both models side-by-side.

---

## Dataset

`data/raw/train.csv` (7,613 rows) and `data/raw/test.csv` (3,263 rows).

Columns: `id`, `keyword` (nullable), `location` (nullable), `text`, `target` (train only, 0/1).
Class split: ~57% non-disaster, ~43% disaster.

---

## Text Preprocessing

Shared pipeline in `src/preprocessing.py` used by both models:
1. Remove URLs, @mentions, # symbols, emojis
2. Lowercase, remove non-alpha characters
3. Tokenize with NLTK `word_tokenize`
4. Remove English stopwords
5. Lemmatize with POS-aware WordNetLemmatizer

---

## Config Files

Both YAML configs follow the same shape:

```yaml
data:
  train_path / test_path / test_size / random_state

model_params:
  (model-specific hyperparameters)

output:
  model_dir / submission_path / figures_dir
```

Edit the config to change hyperparameters without touching code.

---

## Key Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk emoji pyyaml joblib tqdm
pip install torch transformers   # BERT only
```

NLTK data (downloaded automatically on first run):
`stopwords`, `punkt`, `wordnet`, `averaged_perceptron_tagger`
