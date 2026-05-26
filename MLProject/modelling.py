"""MLflow Project training entry point for CI retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mobile JKN sentiment model inside MLflow Project.")
    parser.add_argument("--data-dir", type=Path, default=Path("mobile_jkn_reviews_preprocessing"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=Path("model_output"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("training_artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_data = pd.read_csv(args.data_dir / "train.csv")
    test_data = pd.read_csv(args.data_dir / "test.csv")
    label_mapping = json.loads((args.data_dir / "label_mapping.json").read_text(encoding="utf-8"))

    x_train = train_data["text"].astype(str)
    y_train = train_data["label"]
    x_test = test_data["text"].astype(str)
    y_test = test_data["label"]

    model = build_model(args.random_state)

    with mlflow.start_run() as run:
        mlflow.log_param("dataset_dir", str(args.data_dir))
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("algorithm", "TfidfVectorizer + LogisticRegression")
        mlflow.log_dict(label_mapping, "label_mapping.json")

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.artifact_dir / "classification_report.json"
        matrix_path = args.artifact_dir / "confusion_matrix.json"
        report_path.write_text(
            json.dumps(classification_report(y_test, predictions, output_dict=True, zero_division=0), indent=2),
            encoding="utf-8",
        )
        matrix_path.write_text(
            json.dumps(
                {
                    "labels": list(label_mapping.keys()),
                    "matrix": confusion_matrix(y_test, predictions, labels=list(label_mapping.keys())).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifacts(str(args.artifact_dir), artifact_path="evaluation")

        signature_input = x_train.head(5).tolist()
        signature = infer_signature(signature_input, model.predict(signature_input))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(3).tolist(),
        )

        args.model_output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output / "model.joblib")
        (args.model_output / "label_mapping.json").write_text(
            json.dumps(label_mapping, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifacts(str(args.model_output), artifact_path="serving_model")
        Path("run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        print(f"run_id={run.info.run_id}")
        print(f"accuracy={metrics['accuracy']:.4f}")
        print(f"f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
