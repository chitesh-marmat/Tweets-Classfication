"""
Run inference on a single tweet from the command line.

Usage:
    python -m src.demo "Huge fire breaks out near downtown Seattle"
    python -m src.demo --model bert "flood warning issued for coastal areas"
    python -m src.demo --both "earthquake hits city center, 50 injured"
"""

import argparse
import yaml
import numpy as np

from src.preprocessing import clean_text, download_nltk_resources
from src.utils import get_logger

logger = get_logger(__name__)

LABEL = {0: "NOT DISASTER", 1: "DISASTER"}

MODEL_NAMES = {
    "lr":   "Logistic Regression (TF-IDF)",
    "bert": "DistilBERT (fine-tuned)",
}


def load_config(model: str, config_path: str = None) -> dict:
    path = config_path or f"config/{model}_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def predict_lr(tweet: str, config: dict) -> tuple[str, float]:
    from src.models.lr_model import LogisticRegressionModel
    from src import features

    model = LogisticRegressionModel.load(config["output"]["model_dir"], config)
    cleaned = clean_text(tweet)
    X = features.transform(model.vectorizer, [cleaned])
    proba = model.clf.predict_proba(X)[0]
    label_idx = int(np.argmax(proba))
    return LABEL[label_idx], round(float(proba[label_idx]), 3)


def predict_bert(tweet: str, config: dict) -> tuple[str, float]:
    import torch
    import pandas as pd
    from src.models.bert_model import DistilBertClassifier

    model = DistilBertClassifier.load(config["output"]["model_dir"], config)
    row = pd.DataFrame([{"text": clean_text(tweet), "keyword": "", "location": ""}])
    preds = model.predict(row)
    label_idx = int(preds[0])

    # run one more forward pass to get per-class probabilities
    model.model.eval()
    enc = model.tokenizer(
        clean_text(tweet), return_tensors="pt",
        truncation=True, padding=True, max_length=config["model_params"]["max_length"],
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model.model(**enc).logits
    proba = torch.softmax(logits, dim=1)[0]
    confidence = round(float(proba[label_idx].item()), 3)

    return LABEL[label_idx], confidence


def print_result(model_key: str, label: str, confidence: float):
    bar = "#" * int(confidence * 20)
    print(f"  Model      : {MODEL_NAMES[model_key]}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.1%}  [{bar:<20}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a tweet as disaster or not")
    parser.add_argument("tweet", help="The tweet text to classify")
    parser.add_argument("--model", choices=["lr", "bert"], default="lr",
                        help="Which model to use (default: lr)")
    parser.add_argument("--both", action="store_true",
                        help="Run both models and compare results side by side")
    parser.add_argument("--config", default=None,
                        help="Path to a specific config YAML (ignored when --both is used)")
    args = parser.parse_args()

    download_nltk_resources()

    print(f"\nTweet: \"{args.tweet}\"\n")

    if args.both:
        for key, predictor in [("lr", predict_lr), ("bert", predict_bert)]:
            cfg = load_config(key)
            label, confidence = predictor(args.tweet, cfg)
            print_result(key, label, confidence)
            print()
    else:
        cfg = load_config(args.model, args.config)
        if args.model == "lr":
            label, confidence = predict_lr(args.tweet, cfg)
        else:
            label, confidence = predict_bert(args.tweet, cfg)
        print_result(args.model, label, confidence)
        print()
