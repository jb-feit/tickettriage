"""Baseline classifier: logistic regression on ticket embeddings.

Outputs go to artifacts/baseline/<embedding-tag>/:
  model.joblib   the fitted model plus the metadata needed to use it safely
  metrics.json   numbers to quote in the report
  report.txt     per-intent precision/recall on the frozen test set
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from tickettriage.config import Settings, embedding_tag, get_settings


def _load(s: Settings, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(s.data_dir / "processed" / f"{name}.parquet")
    vecs = np.load(s.data_dir / "embeddings" / embedding_tag(s) / f"{name}.npy")
    if len(df) != len(vecs):
        raise ValueError(f"{name}: {len(df)} rows but {len(vecs)} vectors. Re-run `make embed`.")
    return df, vecs


def _scores(clf: LogisticRegression, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    proba = clf.predict_proba(X)
    pred = clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    routed = conf >= threshold
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mean_confidence": float(conf.mean()),
        # KPI used later on the dashboard: share of tickets confident enough to auto-route.
        "auto_route_rate": float(routed.mean()),
        "accuracy_when_auto_routed": float((pred[routed] == y[routed]).mean())
        if routed.any()
        else None,
    }


def train_baseline(s: Settings, c: float = 10.0, threshold: float = 0.7) -> dict:
    train_df, X_train = _load(s, "train")
    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit(X_train, train_df["intent"])

    metrics: dict = {
        "embedding_tag": embedding_tag(s),
        "n_classes": int(len(clf.classes_)),
        "n_train": int(len(train_df)),
        "C": c,
        "confidence_threshold": threshold,
        "seed": s.seed,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    for split in ("val", "test"):
        df, X = _load(s, split)
        metrics[split] = _scores(clf, X, df["intent"].to_numpy(), threshold)

    # Probe: how does the model treat tickets from intents it has never seen?
    # Lower confidence here is the signal week 4 uses to pick tickets worth labeling.
    future_df, X_future = _load(s, "future")
    if len(future_df):
        conf = clf.predict_proba(X_future).max(axis=1)
        metrics["unseen_intent_probe"] = {
            "n": int(len(conf)),
            "mean_confidence": float(conf.mean()),
            "share_below_threshold": float((conf < threshold).mean()),
        }

    out = s.artifacts_dir / "baseline" / embedding_tag(s)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "labels": list(clf.classes_),
            "embedding_tag": embedding_tag(s),
            "created": metrics["created"],
        },
        out / "model.joblib",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    test_df, X_test = _load(s, "test")
    (out / "report.txt").write_text(
        classification_report(test_df["intent"], clf.predict(X_test), digits=3)
    )
    metrics["output_dir"] = str(out)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the baseline classifier.")
    parser.add_argument("--C", type=float, default=10.0, help="inverse regularisation strength")
    parser.add_argument("--threshold", type=float, default=0.7, help="auto-route confidence")
    args = parser.parse_args(argv)

    m = train_baseline(get_settings(), c=args.C, threshold=args.threshold)
    print(json.dumps({k: v for k, v in m.items() if k != "created"}, indent=2))
    print(f"\nSaved to {Path(m['output_dir'])}")


if __name__ == "__main__":
    main()
