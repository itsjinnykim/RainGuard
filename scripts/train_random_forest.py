from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


matplotlib.use("Agg")


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "rainfall_mm_h",
    "display_expected_stage_max",
    "expected_stage",
    "flood_expected",
    "flood_history_2022",
    "flood_history_2023",
    "flood_history",
    "history_polygon_count",
    "distance_to_flood_area_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest flood-risk classifier from RainGuard grid data."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/flood_dataset.csv",
        help="Path to the generated flood grid dataset.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where model artifacts will be saved.",
    )
    parser.add_argument(
        "--target",
        default="scenario_flood_label",
        help="Target column to predict. Default: scenario_flood_label.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Grouped test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(dataset_path: Path, target_column: str) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run scripts/build_flood_dataset.py first."
        )

    df = pd.read_csv(dataset_path)
    missing_columns = [col for col in FEATURE_COLUMNS + [target_column, "grid_id"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    for column in FEATURE_COLUMNS + [target_column]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    df[target_column] = df[target_column].fillna(0).astype(int)
    return df


def split_by_grid(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    X = df[FEATURE_COLUMNS]
    y = df[target_column]
    groups = df["grid_id"]
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], df.iloc[test_idx]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    metrics_df = pd.DataFrame(
        [{"metric": metric, "value": round(value, 4)} for metric, value in metrics.items()]
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_df = pd.DataFrame(
        cm,
        index=["actual_safe", "actual_risk"],
        columns=["predicted_safe", "predicted_risk"],
    )
    return metrics_df, report_df, confusion_df


def save_feature_importance(model: RandomForestClassifier, model_dir: Path) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df["importance"] = importance_df["importance"].round(6)
    importance_df.to_csv(model_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    plot_df = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#359fa1")
    plt.title("RandomForest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(model_dir / "feature_importance.png", dpi=180)
    plt.close()
    return importance_df


def save_predictions(
    model: RandomForestClassifier,
    test_rows: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_column: str,
    model_dir: Path,
) -> None:
    predictions = test_rows[
        [
            "grid_id",
            "latitude",
            "longitude",
            "rainfall_scenario",
            "rainfall_mm_h",
            "expected_stage",
            "risk_grade",
        ]
    ].copy()
    predictions[f"actual_{target_column}"] = y_test.values
    predictions["predicted_label"] = model.predict(X_test)
    predictions["predicted_risk_probability"] = model.predict_proba(X_test)[:, 1].round(4)
    predictions.to_csv(model_dir / "model_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, args.target)
    X_train, X_test, y_train, y_test, test_rows = split_by_grid(
        df,
        args.target,
        args.test_size,
        args.random_state,
    )

    model = train_model(X_train, y_train, args.random_state)
    metrics_df, report_df, confusion_df = evaluate_model(model, X_test, y_test)
    importance_df = save_feature_importance(model, model_dir)
    save_predictions(model, test_rows, X_test, y_test, args.target, model_dir)

    metrics_df.to_csv(model_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    report_df.to_csv(model_dir / "classification_report.csv", index=False, encoding="utf-8-sig")
    confusion_df.to_csv(model_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    metadata = {
        "dataset": str(dataset_path),
        "target": args.target,
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "model_type": "RandomForestClassifier",
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": args.target,
            "metadata": metadata,
        },
        model_dir / "flood_random_forest.joblib",
    )

    print("Saved model artifacts to:", model_dir)
    print(metrics_df.to_string(index=False))
    print("\nTop feature importance:")
    print(importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
