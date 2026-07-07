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
    "expected_stage",
    "history_polygon_count",
    "history_recent_polygon_count",
    "history_year_count",
    "years_since_latest_history",
    "history_recency_score",
    "distance_to_flood_area_m",
]

DIRECT_LEAKAGE_COLUMNS = {
    "scenario_flood_label",
    "flood_label",
    "scenario_risk_score",
    "risk_grade",
}

LABEL_SOURCE_COLUMNS = {
    "scenario_expected_visible",
    "flood_expected",
    "flood_history",
    "flood_history_2022",
    "flood_history_2023",
    "flood_history_recent",
    "flood_history_old",
}

RISK_SCORE_INPUT_COLUMNS = {
    "rainfall_mm_h",
    "expected_stage",
    "history_polygon_count",
    "history_recent_polygon_count",
    "history_year_count",
    "years_since_latest_history",
    "history_recency_score",
    "distance_to_flood_area_m",
}

SPATIAL_BAND_LABELS = ["west", "mid_west", "mid_east", "east"]


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
    parser.add_argument("--spatial-test-size", type=float, default=0.25, help="East-side spatial holdout ratio.")
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


def split_random_by_grid(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    X = df[FEATURE_COLUMNS]
    y = df[target_column]
    groups = df["grid_id"]
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], df.iloc[test_idx]


def split_spatial_east_west(df: pd.DataFrame, target_column: str, spatial_test_size: float):
    grid_longitudes = df.groupby("grid_id")["longitude"].mean().sort_values()
    split_threshold = grid_longitudes.quantile(1 - spatial_test_size)
    test_grids = set(grid_longitudes[grid_longitudes >= split_threshold].index)

    train_mask = ~df["grid_id"].isin(test_grids)
    test_mask = df["grid_id"].isin(test_grids)

    X = df[FEATURE_COLUMNS]
    y = df[target_column]
    test_rows = df.loc[test_mask].copy()
    test_rows["spatial_split"] = "east_holdout"
    test_rows["spatial_split_threshold_lon"] = split_threshold

    split_info = {
        "method": "east_west_longitude_holdout",
        "test_side": "east",
        "threshold_longitude": round(float(split_threshold), 6),
        "train_grid_count": int(df.loc[train_mask, "grid_id"].nunique()),
        "test_grid_count": int(df.loc[test_mask, "grid_id"].nunique()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }
    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
        test_rows,
        split_info,
    )


def add_spatial_bands(df: pd.DataFrame, band_count: int = 4) -> pd.DataFrame:
    grid_longitudes = df.groupby("grid_id")["longitude"].mean().reset_index()
    labels = SPATIAL_BAND_LABELS if band_count == 4 else [f"band_{index + 1}" for index in range(band_count)]
    grid_longitudes["spatial_band"] = pd.qcut(
        grid_longitudes["longitude"],
        q=band_count,
        labels=labels,
        duplicates="drop",
    ).astype(str)
    return df.merge(grid_longitudes[["grid_id", "spatial_band"]], on="grid_id", how="left")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=12,
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

    if y_test.nunique() > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }
    metrics_df = pd.DataFrame(
        [
            {"metric": metric, "value": round(value, 4) if value is not None else None}
            for metric, value in metrics.items()
        ]
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


def with_split_name(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    named_df = df.copy()
    named_df.insert(0, "validation_split", split_name)
    return named_df


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
    output_path: Path,
) -> None:
    predictions = build_predictions(model, test_rows, X_test, y_test, target_column)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_predictions(
    model: RandomForestClassifier,
    test_rows: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_column: str,
) -> pd.DataFrame:
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
    return predictions


def evaluate_spatial_bands(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
    band_count: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    banded_df = add_spatial_bands(df, band_count)
    metrics_frames = []
    prediction_frames = []
    split_info = []

    for band in sorted(banded_df["spatial_band"].dropna().unique()):
        test_mask = banded_df["spatial_band"] == band
        train_mask = ~test_mask
        X_train = banded_df.loc[train_mask, FEATURE_COLUMNS]
        X_test = banded_df.loc[test_mask, FEATURE_COLUMNS]
        y_train = banded_df.loc[train_mask, target_column]
        y_test = banded_df.loc[test_mask, target_column]
        test_rows = banded_df.loc[test_mask].copy()

        spatial_band_model = train_model(X_train, y_train, random_state)
        metrics_df, _report_df, _confusion_df = evaluate_model(spatial_band_model, X_test, y_test)
        split_name = f"spatial_band_{band}"
        metrics_frames.append(with_split_name(metrics_df, split_name))

        predictions = build_predictions(spatial_band_model, test_rows, X_test, y_test, target_column)
        predictions.insert(0, "validation_split", split_name)
        prediction_frames.append(predictions)

        split_info.append(
            {
                "validation_split": split_name,
                "test_band": band,
                "train_grid_count": int(banded_df.loc[train_mask, "grid_id"].nunique()),
                "test_grid_count": int(banded_df.loc[test_mask, "grid_id"].nunique()),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
        )

    band_metrics_df = pd.concat(metrics_frames, ignore_index=True)
    band_summary_df = (
        band_metrics_df.groupby("metric")["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .round(4)
    )
    band_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return band_metrics_df, band_summary_df, band_predictions_df, split_info


def save_leakage_check(df: pd.DataFrame, target_column: str, model_dir: Path) -> pd.DataFrame:
    checked_columns = sorted(
        set(FEATURE_COLUMNS)
        | DIRECT_LEAKAGE_COLUMNS
        | LABEL_SOURCE_COLUMNS
        | RISK_SCORE_INPUT_COLUMNS
        | {target_column}
    )
    rows = []
    for column in checked_columns:
        in_features = column in FEATURE_COLUMNS
        if column == target_column:
            category = "target"
            status = "pass" if not in_features else "fail"
            note = "Target column must not be used as a feature."
        elif column in DIRECT_LEAKAGE_COLUMNS:
            category = "direct_label_or_score"
            status = "pass" if not in_features else "fail"
            note = "Direct label, risk score, or grade column."
        elif column in LABEL_SOURCE_COLUMNS:
            category = "label_source_binary"
            status = "pass" if not in_features else "review"
            note = "Binary source used in earlier label construction; excluded from v2 features."
        elif column in RISK_SCORE_INPUT_COLUMNS:
            category = "risk_score_input"
            status = "review" if in_features else "pass"
            note = "Public spatial/rainfall feature used to estimate risk; validate with spatial holdout."
        else:
            category = "model_feature"
            status = "pass"
            note = "Feature is not a known direct label column."

        rows.append(
            {
                "column": column,
                "exists_in_dataset": column in df.columns,
                "used_as_feature": in_features,
                "category": category,
                "status": status,
                "note": note,
            }
        )

    leakage_df = pd.DataFrame(rows)
    leakage_df.to_csv(model_dir / "leakage_check.csv", index=False, encoding="utf-8-sig")
    return leakage_df


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, args.target)
    leakage_df = save_leakage_check(df, args.target, model_dir)

    X_train, X_test, y_train, y_test, test_rows = split_random_by_grid(
        df,
        args.target,
        args.test_size,
        args.random_state,
    )

    model = train_model(X_train, y_train, args.random_state)
    metrics_df, report_df, confusion_df = evaluate_model(model, X_test, y_test)
    importance_df = save_feature_importance(model, model_dir)
    save_predictions(model, test_rows, X_test, y_test, args.target, model_dir / "model_predictions.csv")
    save_predictions(model, test_rows, X_test, y_test, args.target, model_dir / "model_predictions_random.csv")

    (
        X_spatial_train,
        X_spatial_test,
        y_spatial_train,
        y_spatial_test,
        spatial_test_rows,
        spatial_split_info,
    ) = split_spatial_east_west(df, args.target, args.spatial_test_size)
    spatial_model = train_model(X_spatial_train, y_spatial_train, args.random_state)
    spatial_metrics_df, spatial_report_df, spatial_confusion_df = evaluate_model(
        spatial_model,
        X_spatial_test,
        y_spatial_test,
    )
    save_predictions(
        spatial_model,
        spatial_test_rows,
        X_spatial_test,
        y_spatial_test,
        args.target,
        model_dir / "model_predictions_spatial.csv",
    )
    (
        spatial_band_metrics_df,
        spatial_band_summary_df,
        spatial_band_predictions_df,
        spatial_band_split_info,
    ) = evaluate_spatial_bands(df, args.target, args.random_state)

    metrics_df.to_csv(model_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(model_dir / "model_metrics_random.csv", index=False, encoding="utf-8-sig")
    spatial_metrics_df.to_csv(model_dir / "model_metrics_spatial.csv", index=False, encoding="utf-8-sig")
    spatial_band_metrics_df.to_csv(
        model_dir / "model_metrics_spatial_bands.csv",
        index=False,
        encoding="utf-8-sig",
    )
    spatial_band_summary_df.to_csv(
        model_dir / "model_metrics_spatial_band_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_df.to_csv(model_dir / "classification_report.csv", index=False, encoding="utf-8-sig")
    with_split_name(report_df, "random_grid").to_csv(
        model_dir / "classification_report_random.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with_split_name(spatial_report_df, "spatial_east_holdout").to_csv(
        model_dir / "classification_report_spatial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confusion_df.to_csv(model_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    confusion_df.to_csv(model_dir / "confusion_matrix_random.csv", encoding="utf-8-sig")
    spatial_confusion_df.to_csv(model_dir / "confusion_matrix_spatial.csv", encoding="utf-8-sig")
    spatial_band_predictions_df.to_csv(
        model_dir / "model_predictions_spatial_bands.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_summary = pd.concat(
        [
            with_split_name(metrics_df, "random_grid"),
            with_split_name(spatial_metrics_df, "spatial_east_holdout"),
            spatial_band_metrics_df,
        ],
        ignore_index=True,
    )
    validation_summary.to_csv(
        model_dir / "model_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "dataset": str(dataset_path),
        "target": args.target,
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "validation_splits": {
            "random_grid": {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_size": args.test_size,
            },
            "spatial_east_holdout": spatial_split_info,
            "spatial_longitude_bands": spatial_band_split_info,
        },
        "direct_leakage_found": bool(
            leakage_df[
                (leakage_df["used_as_feature"])
                & (leakage_df["category"].isin(["target", "direct_label_or_score"]))
            ].shape[0]
        ),
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
    print("\nRandom grid validation:")
    print(metrics_df.to_string(index=False))
    print("\nSpatial east-holdout validation:")
    print(spatial_metrics_df.to_string(index=False))
    print("\nSpatial longitude-band validation summary:")
    print(spatial_band_summary_df.to_string(index=False))
    print("\nLeakage check:")
    print(leakage_df[["column", "used_as_feature", "category", "status"]].to_string(index=False))
    print("\nTop feature importance:")
    print(importance_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
