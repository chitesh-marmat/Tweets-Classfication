"""
Run inference on the test set and write a submission CSV.

Usage:
    python -m src.predict --model lr   --config config/lr_config.yaml
    python -m src.predict --model bert --config config/bert_config.yaml
"""

import argparse
import os
import yaml
import pandas as pd

from src.preprocessing import preprocess_dataframe, download_nltk_resources
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


def predict_lr(config: dict):
    from src.models.lr_model import LogisticRegressionModel

    test_df = pd.read_csv(config["data"]["test_path"])
    logger.info(f"Loaded {len(test_df)} test samples")

    download_nltk_resources()
    test_df = preprocess_dataframe(test_df)

    model = LogisticRegressionModel.load(config["output"]["model_dir"], config)
    preds = model.predict(test_df["text"])

    return test_df["id"], preds


def predict_bert(config: dict):
    from src.models.bert_model import DistilBertClassifier

    test_df = pd.read_csv(config["data"]["test_path"])
    logger.info(f"Loaded {len(test_df)} test samples")

    download_nltk_resources()
    test_df = preprocess_dataframe(test_df)

    model = DistilBertClassifier.load(config["output"]["model_dir"], config)
    preds = model.predict(test_df)

    return test_df["id"], preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for disaster tweet classification")
    parser.add_argument("--model", choices=["lr", "bert"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.model == "lr":
        ids, preds = predict_lr(config)
    else:
        ids, preds = predict_bert(config)

    out_path = config["output"]["submission_path"]
    ensure_dirs(os.path.dirname(out_path))

    submission = pd.DataFrame({"id": ids, "target": preds})
    submission.to_csv(out_path, index=False)
    logger.info(f"Submission saved to {out_path} ({len(submission)} rows)")
