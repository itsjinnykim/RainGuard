import streamlit as st
import folium
import json
import pandas as pd
import re
import warnings
from pathlib import Path
from streamlit_folium import st_folium

warnings.filterwarnings(
    "ignore",
    message="One or several characters couldn't be converted.*",
    category=RuntimeWarning,
)

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import joblib
except ImportError:
    joblib = None


st.set_page_config(
    page_title="RainGuard",
    page_icon="🌧️",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FLOOD_EXPECTED_DIR = DATA_DIR / "flood_expected"
FLOOD_HISTORY_DIR = DATA_DIR / "flood_history"
FLOOD_DATASET_PATH = DATA_DIR / "processed" / "flood_dataset.csv"
AI_GRID_PATH = DATA_DIR / "processed" / "flood_grid.geojson"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "flood_random_forest.joblib"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.csv"

AI_FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "rainfall_mm_h",
    "display_expected_stage_max",
    "expected_stage",
    "flood_expected",
    "flood_history_2022",
    "flood_history_2023",
    "flood_history",
    "flood_history_recent",
    "flood_history_old",
    "history_polygon_count",
    "history_recent_polygon_count",
    "history_old_polygon_count",
    "history_year_count",
    "years_since_latest_history",
    "history_recency_score",
    "distance_to_flood_area_m",
]

RISK_BY_RAINFALL = {
    "10mm/h": {
        "level": "낮음",
        "score": 32,
        "color": "#22c55e",
        "summary": "일부 저지대 주의",
    },
    "30mm/h": {
        "level": "주의",
        "score": 64,
        "color": "#f59e0b",
        "summary": "침수 취약지 위험 상승",
    },
    "50mm/h": {
        "level": "높음",
        "score": 87,
        "color": "#ef4444",
        "summary": "우회 경로 권장",
    },
}

EXPECTED_STAGE_BY_RAINFALL = {
    "10mm/h": 2,
    "30mm/h": 4,
    "50mm/h": 6,
}

EXPECTED_STAGE_STYLE = {
    1: {"label": "~0.5m", "color": "#d7cf2f", "fill": "#fff86a", "opacity": 0.15},
    2: {"label": "0.5~1.0m", "color": "#a9d94a", "fill": "#ccff66", "opacity": 0.16},
    3: {"label": "1.0~1.5m", "color": "#25c189", "fill": "#66f0bd", "opacity": 0.17},
    4: {"label": "1.5~2.0m", "color": "#359fa1", "fill": "#73c9c8", "opacity": 0.18},
    5: {"label": "2.0~3.0m", "color": "#398fca", "fill": "#72c9ff", "opacity": 0.20},
    6: {"label": "3.0m~", "color": "#7e22ce", "fill": "#a855f7", "opacity": 0.22},
}

AI_RISK_GRID_THRESHOLD = 0.5
AI_RISK_STYLE = {
    "낮음": {"color": "#d7cf2f", "fill": "#fff7a3"},
    "주의": {"color": "#73c9c8", "fill": "#b6ece8"},
    "높음": {"color": "#398fca", "fill": "#8fd3ff"},
    "매우 높음": {"color": "#7e22ce", "fill": "#c084fc"},
}

ANALYSIS_BOUNDS = {
    "min_lon": 127.008,
    "min_lat": 37.456,
    "max_lon": 127.124,
    "max_lat": 37.540,
}

RAIN_DROPS = "".join(
    f'<span style="left:{left}%; animation-delay:{delay}s; animation-duration:{duration}s;"></span>'
    for left, delay, duration in [
        (6, 0.0, 2.9),
        (13, 0.8, 3.4),
        (21, 0.3, 3.1),
        (29, 1.1, 3.7),
        (38, 0.5, 3.0),
        (47, 1.5, 3.6),
        (55, 0.2, 3.2),
        (64, 1.0, 3.8),
        (72, 0.6, 3.3),
        (80, 1.7, 3.9),
        (88, 0.9, 3.5),
        (96, 0.4, 3.1),
    ]
)


def find_shp_files(folder):
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.shp"))


def get_expected_stage(shp_path):
    match = re.search(r"DS_FLOODING_(\d+)", shp_path.stem, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def has_required_sidecars(shp_path):
    required = [".shx", ".dbf"]
    return [ext for ext in required if not shp_path.with_suffix(ext).exists()]


def infer_missing_crs(gdf):
    if gdf.empty:
        return "EPSG:4326"

    minx, miny, maxx, maxy = gdf.total_bounds
    if 120 <= minx <= 140 and 30 <= miny <= 45:
        return "EPSG:4326"
    if 100000 <= minx <= 300000 and 400000 <= miny <= 700000:
        return "EPSG:5186"
    if 800000 <= minx <= 1100000 and 1800000 <= miny <= 2100000:
        return "EPSG:5179"
    return "EPSG:5186"


def read_shp_file(shp_path):
    missing = has_required_sidecars(shp_path)
    if missing:
        return None, f"{shp_path.name}: {', '.join(missing)} 파일이 없어 건너뜀"

    read_error = None
    for options in ({}, {"encoding": "utf-8"}, {"encoding": "cp949"}, {"encoding": "euc-kr"}):
        try:
            gdf = gpd.read_file(shp_path, **options)
            break
        except Exception as exc:
            read_error = exc
    else:
        return None, f"{shp_path.name}: 읽기 실패 ({read_error})"

    if "geometry" not in gdf.columns:
        return None, f"{shp_path.name}: geometry 컬럼이 없어 건너뜀"

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None, f"{shp_path.name}: 표시할 geometry가 없음"

    notes = []
    if gdf.crs is None:
        inferred_crs = infer_missing_crs(gdf)
        gdf = gdf.set_crs(inferred_crs, allow_override=True)
        notes.append(f"CRS 없음, {inferred_crs}로 가정")

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return None, f"{shp_path.name}: Polygon/MultiPolygon 데이터가 없음"

    return gdf[["geometry"]].copy(), "; ".join(notes)


def combine_shp_files(shp_files):
    loaded = []
    messages = []
    for shp_path in shp_files:
        gdf, message = read_shp_file(shp_path)
        if message:
            messages.append(message)
        if gdf is not None:
            loaded.append(gdf)

    if not loaded:
        return None, 0, messages

    combined = gpd.GeoDataFrame(
        geometry=pd.concat([item.geometry for item in loaded], ignore_index=True),
        crs="EPSG:4326",
    )
    return combined, len(combined), messages


def combine_gdfs(gdfs):
    valid = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
    if not valid:
        return None

    return gpd.GeoDataFrame(
        geometry=pd.concat([item.geometry for item in valid], ignore_index=True),
        crs="EPSG:4326",
    )


def clip_to_analysis_bounds(gdf):
    if gdf is None or gdf.empty:
        return gdf
    return gdf.cx[
        ANALYSIS_BOUNDS["min_lon"] : ANALYSIS_BOUNDS["max_lon"],
        ANALYSIS_BOUNDS["min_lat"] : ANALYSIS_BOUNDS["max_lat"],
    ].copy()


def simplify_for_map(gdf):
    if gdf is None or gdf.empty:
        return None

    mapped = clip_to_analysis_bounds(gdf)
    if mapped is None or mapped.empty:
        return None

    if len(mapped) > 5000:
        tolerance = 0.00008
    elif len(mapped) > 1000:
        tolerance = 0.00005
    else:
        tolerance = 0.00001

    mapped["geometry"] = mapped.geometry.simplify(tolerance, preserve_topology=True)
    mapped = mapped[mapped.geometry.notna()]
    mapped = mapped[~mapped.geometry.is_empty]
    return mapped


def build_data_signature(load_expected, load_history_2022, load_history_2023, load_history_other):
    shp_files = []
    if load_expected:
        shp_files.extend(find_shp_files(FLOOD_EXPECTED_DIR))

    if load_history_2022 or load_history_2023 or load_history_other:
        history_files = find_shp_files(FLOOD_HISTORY_DIR)
        if load_history_2022:
            shp_files.extend(path for path in history_files if "2022" in path.name)
        if load_history_2023:
            shp_files.extend(path for path in history_files if "2023" in path.name)
        if load_history_other:
            shp_files.extend(
                path for path in history_files if "2022" not in path.name and "2023" not in path.name
            )

    return tuple((str(path), path.stat().st_mtime, path.stat().st_size) for path in shp_files)


@st.cache_data(show_spinner=False)
def load_spatial_layers(signature, load_expected, load_history_2022, load_history_2023, load_history_other):
    if gpd is None:
        return {
            "expected_by_stage": {},
            "history_2022": None,
            "history_2023": None,
            "history_other": None,
            "summary": [
                {"name": "침수예상도", "files": 0, "features": 0},
                {"name": "2022 침수흔적도", "files": 0, "features": 0},
                {"name": "2023 침수흔적도", "files": 0, "features": 0},
                {"name": "과거 침수흔적도", "files": 0, "features": 0},
            ],
            "messages": ["GeoPandas가 설치되어 있지 않아 SHP 파일을 읽을 수 없음"],
        }

    expected_files = find_shp_files(FLOOD_EXPECTED_DIR) if load_expected else []
    history_files = (
        find_shp_files(FLOOD_HISTORY_DIR)
        if load_history_2022 or load_history_2023 or load_history_other
        else []
    )
    history_2022_files = [path for path in history_files if "2022" in path.name] if load_history_2022 else []
    history_2023_files = [path for path in history_files if "2023" in path.name] if load_history_2023 else []
    history_other_files = (
        [path for path in history_files if "2022" not in path.name and "2023" not in path.name]
        if load_history_other
        else []
    )

    expected_by_stage = {}
    expected_stage_counts = {}
    expected_messages = []
    for expected_file in expected_files:
        stage = get_expected_stage(expected_file)
        gdf, message = read_shp_file(expected_file)
        if message:
            expected_messages.append(message)
        if gdf is None:
            continue
        if stage is None:
            stage = len(expected_by_stage) + 1
            expected_messages.append(f"{expected_file.name}: 단계 번호를 파일명에서 찾지 못해 {stage}단계로 표시")
        expected_stage_counts[stage] = len(gdf)
        expected_by_stage[stage] = simplify_for_map(gdf)

    expected_count = sum(expected_stage_counts.values())
    history_2022, history_2022_count, history_2022_messages = combine_shp_files(history_2022_files)
    history_2023, history_2023_count, history_2023_messages = combine_shp_files(history_2023_files)
    history_other, history_other_count, history_other_messages = combine_shp_files(history_other_files)

    messages = []
    if load_expected and not expected_files:
        messages.append("data/flood_expected 폴더에서 .shp 파일을 찾지 못함")
    if load_history_2022 and not history_2022_files:
        messages.append("data/flood_history 폴더에서 2022 .shp 파일을 찾지 못함")
    if load_history_2023 and not history_2023_files:
        messages.append("data/flood_history 폴더에서 2023 .shp 파일을 찾지 못함")

    messages.extend(expected_messages)
    messages.extend(history_2022_messages)
    messages.extend(history_2023_messages)
    messages.extend(history_other_messages)

    return {
        "expected_by_stage": expected_by_stage,
        "expected_stage_counts": expected_stage_counts,
        "history_2022": simplify_for_map(history_2022),
        "history_2023": simplify_for_map(history_2023),
        "history_other": simplify_for_map(history_other),
        "summary": [
            {"name": "침수예상도", "files": len(expected_files), "features": expected_count},
            {"name": "2022 침수흔적도", "files": len(history_2022_files), "features": history_2022_count},
            {"name": "2023 침수흔적도", "files": len(history_2023_files), "features": history_2023_count},
            {"name": "과거 침수흔적도", "files": len(history_other_files), "features": history_other_count},
        ],
        "messages": messages,
    }


def empty_spatial_layers(message="선택된 SHP 레이어 없음"):
    return {
        "expected_by_stage": {},
        "expected_stage_counts": {},
        "history_2022": None,
        "history_2023": None,
        "history_other": None,
        "summary": [
            {"name": "침수예상도", "files": 0, "features": 0},
            {"name": "2022 침수흔적도", "files": 0, "features": 0},
            {"name": "2023 침수흔적도", "files": 0, "features": 0},
            {"name": "과거 침수흔적도", "files": 0, "features": 0},
        ],
        "messages": [message] if message else [],
    }


def build_ai_signature():
    paths = [FLOOD_DATASET_PATH, AI_GRID_PATH, MODEL_PATH, MODEL_METRICS_PATH]
    return tuple(
        (str(path), path.stat().st_mtime, path.stat().st_size)
        for path in paths
        if path.exists()
    )


@st.cache_data(show_spinner=False)
def load_ai_dataset(signature):
    if not FLOOD_DATASET_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FLOOD_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_ai_grid_geojson(signature):
    if not AI_GRID_PATH.exists():
        return None
    with AI_GRID_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_ai_model(signature):
    if joblib is None or not MODEL_PATH.exists():
        return None

    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {
        "model": payload,
        "features": AI_FEATURE_COLUMNS,
        "target": "scenario_flood_label",
    }


@st.cache_data(show_spinner=False)
def load_ai_metrics(signature):
    if not MODEL_METRICS_PATH.exists():
        return {}
    metrics_df = pd.read_csv(MODEL_METRICS_PATH)
    return dict(zip(metrics_df["metric"], metrics_df["value"]))


def format_percent(value, decimals=1):
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}%"


def ai_risk_color(probability):
    return AI_RISK_STYLE[ai_risk_grade(probability)]["color"]


def ai_risk_fill(probability):
    return AI_RISK_STYLE[ai_risk_grade(probability)]["fill"]


def ai_risk_grade(probability):
    if probability >= 0.85:
        return "매우 높음"
    if probability >= 0.68:
        return "높음"
    if probability >= 0.5:
        return "주의"
    return "낮음"


def get_ai_predictions(rainfall):
    signature = build_ai_signature()
    dataset = load_ai_dataset(signature)
    model_payload = load_ai_model(signature)
    metrics = load_ai_metrics(signature)

    if dataset.empty or model_payload is None:
        return pd.DataFrame(), {
            "ready": False,
            "avg_probability": None,
            "risk_cell_count": 0,
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
        }

    scenario_df = dataset[dataset["rainfall_scenario"] == rainfall].copy()
    if scenario_df.empty:
        return pd.DataFrame(), {
            "ready": False,
            "avg_probability": None,
            "risk_cell_count": 0,
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
        }

    features = model_payload.get("features", AI_FEATURE_COLUMNS)
    missing = [feature for feature in features if feature not in scenario_df.columns]
    if missing:
        return pd.DataFrame(), {
            "ready": False,
            "avg_probability": None,
            "risk_cell_count": 0,
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
        }

    model = model_payload["model"]
    X = scenario_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    probabilities = model.predict_proba(X)[:, 1]
    labels = model.predict(X)
    scenario_df["ai_risk_probability"] = probabilities
    scenario_df["ai_predicted_label"] = labels

    return scenario_df, {
        "ready": True,
        "avg_probability": float(probabilities.mean()),
        "risk_cell_count": int(labels.sum()),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
    }


def estimate_grid_step(values, fallback):
    unique_values = sorted(set(round(float(value), 6) for value in values))
    diffs = [
        unique_values[index + 1] - unique_values[index]
        for index in range(len(unique_values) - 1)
        if unique_values[index + 1] - unique_values[index] > 0
    ]
    if not diffs:
        return fallback
    return float(pd.Series(diffs).median())


def build_ai_grid_geojson(prediction_df):
    if prediction_df.empty:
        return None

    visible_df = prediction_df[
        prediction_df["ai_risk_probability"] >= AI_RISK_GRID_THRESHOLD
    ].copy()
    if visible_df.empty:
        return None

    signature = build_ai_signature()
    grid_geojson = load_ai_grid_geojson(signature)
    probability_by_grid = dict(zip(visible_df["grid_id"], visible_df["ai_risk_probability"]))

    if grid_geojson and grid_geojson.get("features"):
        features = []
        for feature in grid_geojson["features"]:
            grid_id = feature.get("properties", {}).get("grid_id")
            probability = probability_by_grid.get(grid_id)
            if probability is None:
                continue

            risk_feature = {
                "type": "Feature",
                "properties": {
                    **feature.get("properties", {}),
                    "probability": float(probability),
                    "probability_text": f"{float(probability) * 100:.0f}%",
                    "risk": ai_risk_grade(float(probability)),
                },
                "geometry": feature["geometry"],
            }
            features.append(risk_feature)

        return {"type": "FeatureCollection", "features": features}

    lat_step = estimate_grid_step(prediction_df["latitude"], 0.0045)
    lon_step = estimate_grid_step(prediction_df["longitude"], 0.0057)

    features = []
    for row in visible_df.itertuples(index=False):
        probability = float(row.ai_risk_probability)
        west = float(row.longitude) - lon_step / 2
        east = float(row.longitude) + lon_step / 2
        south = float(row.latitude) - lat_step / 2
        north = float(row.latitude) + lat_step / 2
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "grid_id": row.grid_id,
                    "probability": probability,
                    "probability_text": f"{probability * 100:.0f}%",
                    "risk": ai_risk_grade(probability),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def add_ai_prediction_layer(map_obj, prediction_df, show):
    if not show:
        return

    grid_geojson = build_ai_grid_geojson(prediction_df)
    if grid_geojson is None:
        return

    folium.GeoJson(
        data=grid_geojson,
        name="AI 위험 격자",
        style_function=lambda feature: {
            "color": ai_risk_color(feature["properties"]["probability"]),
            "weight": 0.7,
            "fillColor": ai_risk_fill(feature["properties"]["probability"]),
            "fillOpacity": min(0.28, 0.08 + feature["properties"]["probability"] * 0.20),
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["probability_text", "risk"],
            aliases=["AI", "Risk"],
            sticky=False,
        ),
        smooth_factor=0.3,
        show=show,
    ).add_to(map_obj)


def add_polygon_layer(map_obj, gdf, name, color, fill_color, fill_opacity, weight, show, dash_array=None):
    if not show:
        return

    if gdf is None or gdf.empty:
        return

    folium.GeoJson(
        data=gdf.to_json(drop_id=True),
        name=name,
        style_function=lambda _feature, color=color, fill_color=fill_color, fill_opacity=fill_opacity, weight=weight: {
            "color": color,
            "weight": weight,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
            "dashArray": dash_array,
        },
        smooth_factor=0.8,
        show=show,
    ).add_to(map_obj)


def render_data_status_card(summary, messages, expected_stage_label, expected_feature_count):
    rows = "".join(
        f"""
        <div class="data-row">
            <span>{item["name"]}</span>
            <b>{item["features"]:,}개</b>
            <small>{item["files"]} file</small>
        </div>
        """
        for item in summary
    )
    notes = "".join(f"<li>{message}</li>" for message in messages[:5])
    notes_html = f"<ul>{notes}</ul>" if notes else "<p>모든 SHP 레이어를 정상적으로 불러왔습니다.</p>"

    st.sidebar.markdown(
        f"""
<div class="sidebar-card">
    <div class="sidebar-card-title">불러온 공간 데이터</div>
    <div class="scenario-row">
        <span>현재 표시 예상도</span>
        <b>{expected_stage_label}</b>
        <small>{expected_feature_count:,}개 polygon</small>
    </div>
    {rows}
    <div class="sidebar-card-note">{notes_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
    .stApp {
        background: #ffffff;
        color: #172033;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: #ffffff;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: #f2f5f9;
        border-right: 1px solid #dbe3ee;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #172033 !important;
    }

    .sidebar-title {
        margin: 16px 0 24px;
    }

    .sidebar-title strong {
        display: block;
        font-size: 25px;
        color: #172033;
        margin-bottom: 0;
    }

    .sidebar-card {
        background: #ffffff;
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .sidebar-card-title {
        color: #172033;
        font-size: 14px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .data-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px 8px;
        padding: 9px 0;
        border-top: 1px solid #edf2f7;
    }

    .data-row:first-of-type {
        border-top: 0;
    }

    .scenario-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px 8px;
        padding: 10px 0 12px;
        margin-bottom: 4px;
        border-bottom: 1px solid #dbeafe;
    }

    .scenario-row span {
        font-size: 13px;
        font-weight: 900;
        color: #0369a1 !important;
    }

    .scenario-row b {
        font-size: 13px;
        color: #0f766e;
    }

    .scenario-row small {
        grid-column: 1 / -1;
        color: #64748b;
        font-size: 12px;
    }

    .data-row span {
        font-size: 13px;
        font-weight: 800;
        color: #334155 !important;
    }

    .data-row b {
        font-size: 13px;
        color: #0f766e;
    }

    .data-row small {
        grid-column: 1 / -1;
        color: #64748b;
        font-size: 12px;
    }

    .sidebar-card-note {
        margin-top: 10px;
        color: #64748b;
        font-size: 12px;
        line-height: 1.55;
    }

    .sidebar-card-note ul {
        margin: 0;
        padding-left: 18px;
    }

    .sidebar-card-note p {
        margin: 0;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 8px;
        padding: 24px 34px;
        margin-bottom: 16px;
        background:
            linear-gradient(110deg, rgba(248, 251, 255, 0.96) 0%, rgba(236, 246, 255, 0.98) 62%, rgba(216, 238, 255, 0.94) 100%);
        border: 1px solid #dbeafe;
        box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 820px;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        padding: 7px 11px;
        border-radius: 999px;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 16px;
    }

    .hero h1 {
        margin: 0;
        color: #1f2937;
        font-size: 44px;
        line-height: 1.04;
        font-weight: 950;
        letter-spacing: 0;
    }

    .hero h1 span {
        color: #0ea5e9;
    }

    .hero p {
        margin: 12px 0 0;
        color: #334155;
        font-size: 18px;
        line-height: 1.45;
        font-weight: 800;
    }

    .hero .small-copy {
        margin-top: 6px;
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.6;
    }

    .rain-field {
        position: absolute;
        top: 0;
        right: 0;
        width: 36%;
        height: 100%;
        z-index: 1;
        pointer-events: none;
        opacity: 0.65;
    }

    .rain-field span {
        position: absolute;
        top: -22px;
        width: 2px;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(14, 165, 233, 0), rgba(14, 165, 233, 0.58));
        animation-name: softRain;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
    }

    @keyframes softRain {
        from {
            transform: translateY(-26px);
            opacity: 0;
        }
        18% {
            opacity: 0.78;
        }
        to {
            transform: translateY(210px);
            opacity: 0;
        }
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 94px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #1f2937;
        font-size: 28px;
        line-height: 1.15;
        font-weight: 950;
    }

    .metric-note {
        color: #64748b;
        font-size: 13px;
        margin-top: 7px;
    }

    .source-note {
        margin-top: 10px;
        color: #64748b;
        font-size: 13px;
        line-height: 1.55;
    }

    .source-note a {
        color: #0369a1;
        font-weight: 800;
        text-decoration: none;
    }

    .map-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-top: 18px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    }

    .section-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-end;
        margin-bottom: 12px;
    }

    .section-head h3 {
        margin: 0;
        color: #1f2937;
        font-size: 20px;
        font-weight: 900;
    }

    .section-head span {
        color: #64748b;
        font-size: 13px;
    }

    iframe {
        border-radius: 8px;
        border: 1px solid #dbe3ee;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #dbe3ee;
        border-radius: 8px;
    }

    @media (max-width: 1100px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 28px 24px;
        }

        .hero h1 {
            font-size: 42px;
        }

        .hero p {
            font-size: 18px;
        }

        .rain-field {
            width: 52%;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    """
<div class="sidebar-title">
    <strong>RainGuard</strong>
</div>
""",
    unsafe_allow_html=True,
)

rainfall = st.sidebar.selectbox(
    "강수량 시나리오 선택",
    ["10mm/h", "30mm/h", "50mm/h"],
)

mode = st.sidebar.radio(
    "경로 추천 기준",
    ["최단경로", "안전경로"],
)

st.sidebar.markdown(
    """
<div style="margin-top:18px; margin-bottom:6px; color:#172033; font-size:14px; font-weight:900;">
    지도 레이어 표시
</div>
""",
    unsafe_allow_html=True,
)

show_expected = st.sidebar.checkbox("침수예상도 표시", value=True)
show_history_2022 = st.sidebar.checkbox("2022 침수흔적도 표시", value=False)
show_history_2023 = st.sidebar.checkbox("2023 침수흔적도 표시", value=False)
show_history_other = st.sidebar.checkbox("과거 침수흔적도 표시", value=False)
show_ai_layer = st.sidebar.checkbox("AI 예측", value=True)
base_map_style = st.sidebar.selectbox(
    "지도 스타일",
    ["CartoDB positron", "OpenStreetMap"],
)

risk = RISK_BY_RAINFALL[rainfall]
if show_ai_layer:
    ai_predictions, ai_summary = get_ai_predictions(rainfall)
else:
    ai_predictions = pd.DataFrame()
    ai_summary = {
        "ready": False,
        "avg_probability": None,
        "risk_cell_count": 0,
        "recall": None,
        "f1_score": None,
        "roc_auc": None,
    }
ai_avg_text = format_percent(ai_summary["avg_probability"], 1)
ai_recall_text = format_percent(ai_summary["recall"], 1)
ai_cells_text = f"{ai_summary['risk_cell_count']:,}개" if ai_summary["ready"] else "-"
ai_color = ai_risk_color(ai_summary["avg_probability"] or 0)

max_expected_stage = EXPECTED_STAGE_BY_RAINFALL[rainfall]
spatial_layer_requested = show_expected or show_history_2022 or show_history_2023 or show_history_other
if spatial_layer_requested:
    with st.spinner("SHP 공간 데이터를 불러오는 중입니다..."):
        spatial_layers = load_spatial_layers(
            build_data_signature(show_expected, show_history_2022, show_history_2023, show_history_other),
            show_expected,
            show_history_2022,
            show_history_2023,
            show_history_other,
        )
else:
    spatial_layers = empty_spatial_layers()

visible_expected_stages = (
    [
        stage
        for stage in sorted(spatial_layers["expected_by_stage"])
        if stage <= max_expected_stage
    ]
    if show_expected
    else []
)
visible_expected_count = sum(
    spatial_layers["expected_stage_counts"].get(stage, 0)
    for stage in visible_expected_stages
)
expected_stage_label = f"1~{max_expected_stage}단계" if show_expected else "꺼짐"

st.sidebar.markdown(
    f"""
<div style="margin-top:22px; color:#334155; font-size:15px; line-height:1.7;">
    선택한 강수량: <b>{rainfall}</b><br>
    표시 예상도: <b>{expected_stage_label}</b><br>
    위험도: <b style="color:{risk['color']}">{risk['level']}</b>
</div>
""",
    unsafe_allow_html=True,
)

render_data_status_card(
    spatial_layers["summary"],
    spatial_layers["messages"],
    expected_stage_label,
    visible_expected_count,
)


st.markdown(
    f"""
<section class="hero">
    <div class="rain-field">{RAIN_DROPS}</div>
    <div class="hero-content">
        <h1>Rain<span>Guard</span></h1>
        <p>공공데이터 기반 도시 침수 위험 예측 서비스</p>
        <div class="small-copy">폭우 시 빠른 길보다 안전한 길을 찾기 위한 침수 위험 지도</div>
    </div>
</section>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">분석 지역</div>
        <div class="metric-value">강남구</div>
        <div class="metric-note">SHP + AI</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">강수량</div>
        <div class="metric-value">{rainfall}</div>
        <div class="metric-note">{expected_stage_label}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">AI 위험확률</div>
        <div class="metric-value" style="color:{ai_color}">{ai_avg_text}</div>
        <div class="metric-note">평균 예측값</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">위험 격자</div>
        <div class="metric-value">{ai_cells_text}</div>
        <div class="metric-note">Recall {ai_recall_text}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


analysis_center = [37.5172, 127.0473]

m = folium.Map(
    location=analysis_center,
    zoom_start=13,
    tiles=base_map_style,
    prefer_canvas=True,
)

if base_map_style == "CartoDB positron":
    tile_tone_css = """
    <style>
        .leaflet-tile-pane img {
            filter: contrast(1.08) saturate(1.05) brightness(0.98);
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(tile_tone_css))

for stage in visible_expected_stages:
    style = EXPECTED_STAGE_STYLE.get(stage, EXPECTED_STAGE_STYLE[6])
    add_polygon_layer(
        m,
        spatial_layers["expected_by_stage"].get(stage),
        f"침수예상도 {style['label']}",
        color=style["color"],
        fill_color=style["fill"],
        fill_opacity=style["opacity"],
        weight=0.9,
        show=show_expected,
    )

add_polygon_layer(
    m,
    spatial_layers["history_2022"],
    "2022 침수흔적도",
    color="#d97706",
    fill_color="#facc15",
    fill_opacity=0.18,
    weight=1.6,
    show=show_history_2022,
)

add_polygon_layer(
    m,
    spatial_layers["history_2023"],
    "2023 침수흔적도",
    color="#ef4444",
    fill_color="#fb7185",
    fill_opacity=0.2,
    weight=1.8,
    show=show_history_2023,
    dash_array="4",
)

add_polygon_layer(
    m,
    spatial_layers["history_other"],
    "과거 침수흔적도",
    color="#475569",
    fill_color="#94a3b8",
    fill_opacity=0.12,
    weight=1.1,
    show=show_history_other,
    dash_array="2",
)

add_ai_prediction_layer(m, ai_predictions, show_ai_layer)

folium.Marker(
    location=analysis_center,
    popup="RainGuard 분석 중심",
    tooltip="분석 중심",
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

expected_legend_rows = "".join(
    f"""
    <div style="display:flex; align-items:center; gap:7px; margin-top:4px;">
        <span style="display:inline-block;width:18px;height:10px;background:{style['fill']};border:1px solid {style['color']};"></span>
        <span>{style['label']}</span>
    </div>
"""
    for stage, style in EXPECTED_STAGE_STYLE.items()
    if stage <= max_expected_stage
)

legend_html = f"""
<div style="
    position: fixed;
    right: 28px;
    bottom: 28px;
    z-index: 9999;
    background: rgba(255,255,255,0.94);
    border: 1px solid #dbe3ee;
    border-radius: 8px;
    padding: 10px 12px;
    color: #172033;
    font-size: 12px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.14);
">
    <div style="font-weight: 900; margin-bottom: 6px;">SHP 침수심</div>
    {expected_legend_rows}
    <div style="height:1px;background:#e2e8f0;margin:9px 0 7px;"></div>
    <div style="font-weight: 900; margin-bottom: 5px;">AI Grid</div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#b6ece8;border:1px solid #73c9c8;margin-right:6px;"></span>50%+</div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#8fd3ff;border:1px solid #398fca;margin-right:6px;"></span>68%+</div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#c084fc;border:1px solid #7e22ce;margin-right:6px;"></span>85%+</div>
    <div style="height:1px;background:#e2e8f0;margin:9px 0 7px;"></div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#facc15;border:1px solid #d97706;margin-right:6px;"></span>2022 침수흔적도</div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#fb7185;border:1px solid #ef4444;margin-right:6px;"></span>2023 침수흔적도</div>
    <div><span style="display:inline-block;width:18px;height:10px;background:#94a3b8;border:1px solid #475569;margin-right:6px;"></span>과거 침수흔적도</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st.markdown(
    f"""
<div class="map-card">
    <div class="section-head">
        <h3>RainGuard Map</h3>
        <span>{rainfall} · AI + SHP</span>
    </div>
    <div class="source-note">
        AI 위험 격자 + SHP 레이어
    </div>
""",
    unsafe_allow_html=True,
)

map_key = (
    f"rainguard_map_stable_{base_map_style}_{rainfall}_{show_expected}_"
    f"{show_history_2022}_{show_history_2023}_{show_history_other}_"
    f"{show_ai_layer}_{expected_stage_label}"
)
st_folium(m, width=1240, height=620, key=map_key, returned_objects=[])

st.markdown("</div>", unsafe_allow_html=True)
