import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted"), 4),
    }


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=["Not Disaster", "Disaster"]))


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Disaster", "Disaster"],
        yticklabels=["Not Disaster", "Disaster"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_names, coefficients, top_n: int = 20, title: str = "Top Features", save_path: str = None):
    # take top_n features by absolute weight
    indices = np.argsort(np.abs(coefficients))[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_weights = [coefficients[i] for i in indices]

    colors = ["steelblue" if w > 0 else "salmon" for w in top_weights]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features, top_weights, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Coefficient weight")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tweet_length_distribution(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # overall length distribution
    df["text"].str.len().hist(bins=40, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Tweet Length Distribution")
    axes[0].set_xlabel("Character count")
    axes[0].set_ylabel("Frequency")

    # length by class (only if target column exists)
    if "target" in df.columns:
        for label, group in df.groupby("target"):
            group["text"].str.len().hist(bins=40, ax=axes[1], alpha=0.6, label=f"Class {label}", edgecolor="white")
        axes[1].set_title("Tweet Length by Class")
        axes[1].set_xlabel("Character count")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_top_keywords(df: pd.DataFrame, top_n: int = 15, save_path: str = None):
    if "keyword" not in df.columns or "target" not in df.columns:
        return

    keyword_counts = df[df["keyword"] != ""].groupby(["keyword", "target"]).size().unstack(fill_value=0)
    keyword_counts["total"] = keyword_counts.sum(axis=1)
    top = keyword_counts.nlargest(top_n, "total").drop(columns="total")

    top.plot(kind="barh", figsize=(9, 6), color=["steelblue", "salmon"])
    plt.title(f"Top {top_n} Keywords by Class")
    plt.xlabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
