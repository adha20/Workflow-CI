"""MLflow Project entry point for hotel booking cancellation CI retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABELS = [0, 1]
LABEL_NAMES = ["not_canceled", "canceled"]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(data_dir: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict, dict, list[str]]:
    train_data = pd.read_csv(data_dir / "train.csv")
    test_data = pd.read_csv(data_dir / "test.csv")
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    label_mapping = load_json(data_dir / "label_mapping.json", {"0": "not_canceled", "1": "canceled"})
    dataset_info = load_json(data_dir / "dataset_info.json", {})
    feature_columns = load_json(data_dir / "feature_columns.json", list(x_train.columns))
    return x_train, y_train, x_test, y_test, label_mapping, dataset_info, feature_columns


def build_model(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        max_features=0.6,
        min_samples_leaf=2,
        min_samples_split=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def feature_importance(model: RandomForestClassifier, feature_names: list[str], limit: int = 50) -> pd.DataFrame:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:limit]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "feature": [feature_names[index] for index in order],
            "importance": [float(importances[index]) for index in order],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hotel booking cancellation model inside MLflow Project.")
    parser.add_argument("--data-dir", type=Path, default=Path("hotel_bookings_preprocessing"))
    parser.add_argument("--target-column", default="is_canceled")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=Path("model_output"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("training_artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, y_test, label_mapping, dataset_info, feature_columns = load_dataset(
        args.data_dir,
        args.target_column,
    )
    model = build_model(args.random_state)

    with mlflow.start_run() as run:
        mlflow.log_param("dataset_dir", str(args.data_dir))
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))
        mlflow.log_param("feature_count", x_train.shape[1])

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        probability_matrix = model.predict_proba(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "log_loss": log_loss(y_test, probability_matrix, labels=LABELS),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.artifact_dir / "metrics.json", metrics)
        write_json(args.artifact_dir / "classification_report.json", classification_report(y_test, predictions, target_names=LABEL_NAMES, output_dict=True, zero_division=0))
        write_json(args.artifact_dir / "confusion_matrix.json", {"labels": LABEL_NAMES, "matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist()})
        feature_importance(model, list(x_train.columns)).to_csv(args.artifact_dir / "feature_importance.csv", index=False)
        pd.DataFrame(
            {
                "actual": y_test.head(200).to_numpy(),
                "prediction": predictions[:200],
                "cancellation_probability": probabilities[:200],
            }
        ).to_csv(args.artifact_dir / "sample_predictions.csv", index=False)
        mlflow.log_dict(label_mapping, "label_mapping.json")
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_dict({"feature_columns": feature_columns}, "feature_columns.json")
        mlflow.log_artifacts(str(args.artifact_dir), artifact_path="evaluation")

        signature = infer_signature(x_train.head(5), model.predict(x_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(3),
        )

        args.model_output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output / "model.joblib")
        write_json(args.model_output / "label_mapping.json", label_mapping)
        write_json(args.model_output / "feature_columns.json", feature_columns)
        mlflow.log_artifacts(str(args.model_output), artifact_path="serving_model")
        Path("run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        print(f"run_id={run.info.run_id}")
        print(f"accuracy={metrics['accuracy']:.4f}")
        print(f"f1={metrics['f1']:.4f}")
        print(f"roc_auc={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
