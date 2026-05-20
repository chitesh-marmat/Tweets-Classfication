"""
Train a model on the disaster tweets dataset.

Usage:
    python -m src.train --model lr   --config config/lr_config.yaml
    python -m src.train --model bert --config config/bert_config.yaml
"""

import argparse
import os
import yaml
import pandas as pd

from src.preprocessing import preprocess_dataframe, download_nltk_resources
from src.utils import set_seed, get_logger, ensure_dirs

logger = get_logger(__name__)


def train_lr(config: dict):
    from src.models.lr_model import LogisticRegressionModel
    from src import evaluate

    set_seed(config["data"]["random_state"])
    download_nltk_resources()

    df = pd.read_csv(config["data"]["train_path"])
    logger.info(f"Loaded {len(df)} training samples")

    df = preprocess_dataframe(df)

    model = LogisticRegressionModel(config)
    y_true, y_pred = model.fit(df)

    # save model
    model.save(config["output"]["model_dir"])

    # save evaluation plots
    fig_dir = config["output"]["figures_dir"]
    ensure_dirs(fig_dir)

    evaluate.plot_confusion_matrix(
        y_true, y_pred,
        title="LR — Confusion Matrix",
        save_path=os.path.join(fig_dir, "confusion_matrix.png"),
    )
    evaluate.plot_feature_importance(
        model.get_feature_names(),
        model.get_coefficients(),
        top_n=20,
        title="LR — Top 20 Features",
        save_path=os.path.join(fig_dir, "feature_importance.png"),
    )
    logger.info(f"Plots saved to {fig_dir}")


def train_bert(config: dict):
    from src.models.bert_model import DistilBertClassifier
    from src import evaluate

    set_seed(config["data"]["random_state"])
    download_nltk_resources()

    df = pd.read_csv(config["data"]["train_path"])
    logger.info(f"Loaded {len(df)} training samples")

    df = preprocess_dataframe(df)

    model = DistilBertClassifier(config)
    y_true, y_pred = model.fit(df)

    model.save(config["output"]["model_dir"])

    fig_dir = config["output"]["figures_dir"]
    ensure_dirs(fig_dir)

    evaluate.plot_confusion_matrix(
        y_true, y_pred,
        title="DistilBERT — Confusion Matrix",
        save_path=os.path.join(fig_dir, "confusion_matrix.png"),
    )
    logger.info(f"Plots saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tweet disaster classifier")
    parser.add_argument("--model", choices=["lr", "bert"], required=True, help="Which model to train")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ensure_dirs(
        config["output"]["model_dir"],
        config["output"]["figures_dir"],
        os.path.dirname(config["output"]["submission_path"]),
    )

    if args.model == "lr":
        train_lr(config)
    else:
        train_bert(config)
