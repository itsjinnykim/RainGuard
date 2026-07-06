from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


BACKGROUND = "#f6f8fb"
TEXT = "#172033"
MUTED = "#64748b"
LINE = "#dbe3ee"
TEAL = "#359fa1"
NAVY = "#1f2a44"
BLUE = "#398fca"
PURPLE = "#7e22ce"
YELLOW = "#d7cf2f"

METRIC_LABELS = {
    "accuracy": ("Accuracy", "전체 예측 중 맞힌 비율"),
    "precision": ("Precision", "위험 예측 중 실제 위험 비율"),
    "recall": ("Recall", "실제 위험지역을 놓치지 않은 비율"),
    "f1_score": ("F1-score", "Precision과 Recall의 균형"),
    "roc_auc": ("ROC-AUC", "위험/안전 구분 성능"),
}

FEATURE_LABELS = {
    "distance_to_flood_area_m": "침수지역까지 거리",
    "expected_stage": "침수예상도 단계",
    "flood_expected": "침수예상도 포함 여부",
    "history_polygon_count": "과거 침수 polygon 수",
    "flood_history_2022": "2022 침수 이력",
    "flood_history": "과거 침수 이력",
    "flood_history_recent": "최근 침수 이력",
    "flood_history_old": "오래된 침수 이력",
    "history_recent_polygon_count": "최근 침수 polygon 수",
    "history_old_polygon_count": "오래된 침수 polygon 수",
    "history_year_count": "침수 발생 연도 수",
    "years_since_latest_history": "마지막 침수 이후 기간",
    "history_recency_score": "최근성 가중 이력",
    "latitude": "위도",
    "longitude": "경도",
    "rainfall_mm_h": "강수량",
    "display_expected_stage_max": "표시 예상도 단계",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PPT-ready model performance images.")
    parser.add_argument("--model-dir", default="models", help="Directory containing model CSV files.")
    parser.add_argument(
        "--output",
        default="models/model_metrics_ppt.png",
        help="PNG path to write the PPT-ready metric dashboard.",
    )
    return parser.parse_args()


def set_korean_font() -> None:
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        try:
            font_path = fm.findfont(font_name, fallback_to_default=False)
        except ValueError:
            continue
        if font_path:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_metrics(model_dir: Path) -> dict[str, float]:
    metrics_df = pd.read_csv(model_dir / "model_metrics.csv")
    return dict(zip(metrics_df["metric"], metrics_df["value"]))


def read_confusion(model_dir: Path) -> pd.DataFrame:
    return pd.read_csv(model_dir / "confusion_matrix.csv", index_col=0)


def read_importance(model_dir: Path) -> pd.DataFrame:
    importance_df = pd.read_csv(model_dir / "feature_importance.csv").head(5).copy()
    importance_df["feature_label"] = importance_df["feature"].map(FEATURE_LABELS).fillna(
        importance_df["feature"]
    )
    return importance_df


def rounded_box(ax, xy, width, height, radius=0.035, facecolor="white", edgecolor=LINE, linewidth=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def add_metric_card(ax, x, y, w, h, metric_key, value, color, highlight=False):
    label, description = METRIC_LABELS[metric_key]
    value_text = f"{value * 100:.2f}%" if metric_key == "roc_auc" else f"{value * 100:.1f}%"
    face = "#ffffff" if not highlight else "#eefafa"
    edge = color if highlight else LINE
    rounded_box(ax, (x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6 if highlight else 1.0)

    ax.text(x + 0.04 * w, y + h - 0.25 * h, label, fontsize=18, fontweight="bold", color=TEXT)
    ax.text(x + 0.04 * w, y + h - 0.44 * h, description, fontsize=9.5, color=MUTED)
    ax.text(
        x + 0.04 * w,
        y + 0.20 * h,
        value_text,
        fontsize=31 if highlight else 27,
        fontweight="bold",
        color=color,
    )


def add_confusion_matrix(ax, confusion_df: pd.DataFrame):
    x, y, w, h = 0.57, 0.15, 0.36, 0.26
    rounded_box(ax, (x, y), w, h)
    ax.text(x + 0.03 * w, y + h - 0.16 * h, "Confusion Matrix", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(x + 0.03 * w, y + h - 0.31 * h, "테스트 데이터 기준 예측 결과", fontsize=10, color=MUTED)

    safe_safe = int(confusion_df.loc["actual_safe", "predicted_safe"])
    safe_risk = int(confusion_df.loc["actual_safe", "predicted_risk"])
    risk_safe = int(confusion_df.loc["actual_risk", "predicted_safe"])
    risk_risk = int(confusion_df.loc["actual_risk", "predicted_risk"])

    cell_x = x + 0.08 * w
    cell_y = y + 0.08 * h
    cell_w = 0.19 * w
    cell_h = 0.21 * h
    values = [
        ("안전→안전", safe_safe, "#e0f2fe"),
        ("안전→위험", safe_risk, "#fef3c7"),
        ("위험→안전", risk_safe, "#fee2e2"),
        ("위험→위험", risk_risk, "#dcfce7"),
    ]
    for idx, (label, value, color) in enumerate(values):
        col = idx % 2
        row = idx // 2
        px = cell_x + col * (cell_w + 0.025 * w)
        py = cell_y + (1 - row) * (cell_h + 0.035 * h)
        rounded_box(ax, (px, py), cell_w, cell_h, radius=0.018, facecolor=color, edgecolor="white", linewidth=0)
        ax.text(px + 0.08 * cell_w, py + 0.58 * cell_h, label, fontsize=9.5, color=MUTED)
        ax.text(px + 0.08 * cell_w, py + 0.16 * cell_h, f"{value}", fontsize=19, fontweight="bold", color=TEXT)


def add_feature_importance(ax, importance_df: pd.DataFrame):
    x, y, w, h = 0.07, 0.15, 0.45, 0.26
    rounded_box(ax, (x, y), w, h)
    ax.text(x + 0.03 * w, y + h - 0.16 * h, "Feature Importance", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(x + 0.03 * w, y + h - 0.31 * h, "모델이 위험 판단에 크게 사용한 변수", fontsize=10, color=MUTED)

    max_importance = importance_df["importance"].max()
    bar_x = x + 0.04 * w
    bar_y = y + 0.08 * h
    bar_w = 0.72 * w
    row_h = 0.105 * h
    colors = [TEAL, BLUE, PURPLE, YELLOW, "#94a3b8"]
    display_df = importance_df.reset_index(drop=True)
    for rank, row in display_df.iterrows():
        py = bar_y + (len(display_df) - rank - 1) * row_h
        label = row["feature_label"]
        value = row["importance"]
        ax.text(bar_x, py + 0.025, label, fontsize=9.3, color=TEXT, va="center")
        ax.add_patch(Rectangle((bar_x + 0.40 * w, py), bar_w * 0.48, 0.018, color="#e2e8f0", linewidth=0))
        ax.add_patch(
            Rectangle(
                (bar_x + 0.40 * w, py),
                bar_w * 0.48 * (value / max_importance),
                0.018,
                color=colors[rank % len(colors)],
                linewidth=0,
            )
        )
        ax.text(bar_x + 0.76 * w, py + 0.009, f"{value:.3f}", fontsize=9.0, color=MUTED, va="center")


def make_dashboard(metrics: dict[str, float], confusion_df: pd.DataFrame, importance_df: pd.DataFrame, output: Path):
    set_korean_font()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.06, 0.91, "RainGuard AI Model Performance", fontsize=31, fontweight="bold", color=TEXT)
    ax.text(
        0.06,
        0.865,
        "RandomForest 기반 침수 위험 예측 모델  |  공공데이터 SHP 격자 데이터셋 검증 결과",
        fontsize=13,
        color=MUTED,
    )

    rounded_box(ax, (0.06, 0.77), 0.88, 0.055, radius=0.025, facecolor="#eaf7f7", edgecolor="#ccebec", linewidth=1.0)
    ax.text(
        0.085,
        0.79,
        "핵심 지표: 침수 예측에서는 위험지역을 놓치지 않는 것이 중요하므로 Recall을 주요 성능 지표로 사용",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )

    card_keys = ["accuracy", "recall", "f1_score", "roc_auc"]
    card_colors = [BLUE, TEAL, PURPLE, NAVY]
    x0, y0, w, h, gap = 0.06, 0.50, 0.205, 0.20, 0.02
    for i, key in enumerate(card_keys):
        add_metric_card(
            ax,
            x0 + i * (w + gap),
            y0,
            w,
            h,
            key,
            metrics[key],
            card_colors[i],
            highlight=(key == "recall"),
        )

    add_feature_importance(ax, importance_df)
    add_confusion_matrix(ax, confusion_df)

    ax.text(
        0.06,
        0.075,
        "Source: Seoul flood expected map + 2022/2023 flood trace SHP, RainGuard grid dataset",
        fontsize=9.5,
        color="#94a3b8",
    )
    ax.text(
        0.94,
        0.075,
        "Generated from model_metrics.csv",
        fontsize=9.5,
        color="#94a3b8",
        ha="right",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    metrics = read_metrics(model_dir)
    confusion_df = read_confusion(model_dir)
    importance_df = read_importance(model_dir)
    make_dashboard(metrics, confusion_df, importance_df, output)
    print(f"Saved PPT-ready model metric image: {output}")


if __name__ == "__main__":
    main()
