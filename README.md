# Disaster Tweet Classification

Binary classification of tweets — does this tweet describe a real disaster?  
Dataset: [Kaggle NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) (7,613 train / 3,263 test tweets).

Two models are compared:

| Model | Val Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~81% | TF-IDF (5000 features) + sklearn LR |
| DistilBERT | ~80.7% | Fine-tuned `distilbert-base-uncased`, 3 epochs |

---

## Project Structure

```
├── src/
│   ├── preprocessing.py   # shared text cleaning pipeline
│   ├── features.py        # TF-IDF helpers
│   ├── evaluate.py        # metrics + plot functions
│   ├── utils.py           # seed, logger, paths
│   ├── train.py           # CLI — train a model
│   ├── predict.py         # CLI — run inference
│   └── models/
│       ├── lr_model.py    # LogisticRegressionModel
│       └── bert_model.py  # DistilBertClassifier
├── config/
│   ├── lr_config.yaml     # LR hyperparameters
│   └── bert_config.yaml   # BERT hyperparameters
├── notebooks/
│   ├── 01_eda.ipynb       # data exploration
│   └── 02_experiments.ipynb  # interactive training + comparison
├── data/raw/              # train.csv, test.csv
├── models/                # saved model artifacts (gitignored)
└── outputs/               # figures + submission CSVs (gitignored)
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

### Train

```bash
# Logistic Regression (~10 seconds)
python -m src.train --model lr --config config/lr_config.yaml

# DistilBERT (~15 min on CPU, ~3 min on GPU)
python -m src.train --model bert --config config/bert_config.yaml
```

After training:
- model files are saved to `models/lr/` or `models/bert/`
- confusion matrix + feature importance plots saved to `outputs/figures/`

### Predict (generate submission CSV)

```bash
python -m src.predict --model lr   --config config/lr_config.yaml
python -m src.predict --model bert --config config/bert_config.yaml
```

Submission files land in `outputs/submissions/`.

### Notebooks

```bash
cd notebooks && jupyter notebook
```

- `01_eda.ipynb` — data exploration, class distribution, keyword analysis
- `02_experiments.ipynb` — train both models interactively, compare metrics side-by-side

---

## Text Preprocessing

Both models use the same cleaning pipeline (`src/preprocessing.py`):
1. Remove URLs, @mentions, `#` symbols, emojis
2. Lowercase, strip non-alpha characters
3. Tokenize (NLTK `word_tokenize`)
4. Remove English stopwords
5. POS-aware lemmatization (WordNetLemmatizer)

NLTK resources are downloaded automatically on first run.

---

## Changing Hyperparameters

Edit `config/lr_config.yaml` or `config/bert_config.yaml` — no code changes needed.

Key knobs:

```yaml
# lr_config.yaml
features:
  max_features: 5000      # vocabulary size for TF-IDF
model_params:
  C: 1.0                  # regularization strength

# bert_config.yaml
model_params:
  epochs: 3
  batch_size: 16
  lr: 5.0e-5
```
