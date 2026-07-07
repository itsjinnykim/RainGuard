import streamlit as st
import folium
import json
import math
import pandas as pd
import pickle
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

try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None

try:
    import networkx as nx
    import osmnx as ox
except ImportError:
    nx = None
    ox = None


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
ROAD_GRAPH_PATH = DATA_DIR / "road_network" / "gangnam_walk.graphml"
ROAD_GRAPH_PICKLE_PATH = DATA_DIR / "road_network" / "gangnam_walk.pkl"

CLICK_ROUTE_MIN_DISTANCE_KM = 0.08
CLICK_ROUTE_MAX_NODE_DISTANCE_M = 900

AI_FEATURE_COLUMNS = [
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


def build_road_graph_signature():
    graph_path = ROAD_GRAPH_PICKLE_PATH if ROAD_GRAPH_PICKLE_PATH.exists() else ROAD_GRAPH_PATH
    if not graph_path.exists():
        return None
    return (str(graph_path), graph_path.stat().st_mtime, graph_path.stat().st_size)


@st.cache_resource(show_spinner=False)
def load_road_graph(signature):
    if signature is None or ox is None or nx is None:
        return None
    graph_path = Path(signature[0])
    if graph_path.suffix.lower() == ".pkl":
        with graph_path.open("rb") as f:
            return pickle.load(f)
    return ox.load_graphml(graph_path)


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


def get_route_points():
    return {
        "신사역": {"name": "신사역", "lat": 37.5163, "lon": 127.0200},
        "논현역": {"name": "논현역", "lat": 37.5111, "lon": 127.0214},
        "신논현역": {"name": "신논현역", "lat": 37.5046, "lon": 127.0250},
        "강남역": {"name": "강남역", "lat": 37.4979, "lon": 127.0276},
        "역삼역": {"name": "역삼역", "lat": 37.5007, "lon": 127.0365},
        "선릉역": {"name": "선릉역", "lat": 37.5045, "lon": 127.0490},
        "삼성역": {"name": "삼성역", "lat": 37.5088, "lon": 127.0632},
        "봉은사역": {"name": "봉은사역", "lat": 37.5142, "lon": 127.0602},
        "청담역": {"name": "청담역", "lat": 37.5194, "lon": 127.0530},
        "강남구청역": {"name": "강남구청역", "lat": 37.5172, "lon": 127.0413},
        "압구정로데오역": {"name": "압구정로데오역", "lat": 37.5275, "lon": 127.0406},
        "선정릉역": {"name": "선정릉역", "lat": 37.5110, "lon": 127.0436},
        "한티역": {"name": "한티역", "lat": 37.4963, "lon": 127.0529},
        "도곡역": {"name": "도곡역", "lat": 37.4909, "lon": 127.0555},
        "대치역": {"name": "대치역", "lat": 37.4945, "lon": 127.0632},
        "학여울역": {"name": "학여울역", "lat": 37.4967, "lon": 127.0706},
        "대청역": {"name": "대청역", "lat": 37.4936, "lon": 127.0795},
        "일원역": {"name": "일원역", "lat": 37.4837, "lon": 127.0844},
        "수서역": {"name": "수서역", "lat": 37.4875, "lon": 127.1015},
        "세곡동 주민센터": {"name": "세곡동 주민센터", "lat": 37.4690, "lon": 127.1068},
        "강남세브란스병원": {"name": "강남세브란스병원", "lat": 37.4928, "lon": 127.0463},
        "삼성서울병원": {"name": "삼성서울병원", "lat": 37.4883, "lon": 127.0851},
        "코엑스": {"name": "코엑스", "lat": 37.5118, "lon": 127.0592},
    }


def calculate_distance_km(start, end):
    earth_radius_km = 6371.0
    start_lat = math.radians(start["lat"])
    start_lon = math.radians(start["lon"])
    end_lat = math.radians(end["lat"])
    end_lon = math.radians(end["lon"])
    d_lat = end_lat - start_lat
    d_lon = end_lon - start_lon

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(start_lat) * math.cos(end_lat) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(earth_radius_km * c, 2)


def format_route_point_signature(point):
    if not point:
        return "none"
    return f"{point['lat']:.6f},{point['lon']:.6f}"


def make_clicked_route_point(label, coordinate):
    if not coordinate:
        return None
    return {
        "name": label,
        "lat": float(coordinate["lat"]),
        "lon": float(coordinate["lon"]),
        "tooltip": label,
    }


def nearest_graph_node_with_distance(graph, point):
    node = ox.distance.nearest_nodes(graph, X=point["lon"], Y=point["lat"])
    node_data = graph.nodes[node]
    node_point = {"lat": float(node_data["y"]), "lon": float(node_data["x"])}
    distance_m = calculate_distance_km(point, node_point) * 1000
    return node, distance_m


def initialize_click_route_state():
    defaults = {
        "clicked_start": None,
        "clicked_end": None,
        "clicked_last_signature": None,
        "clicked_route_results": None,
        "clicked_route_context": None,
        "clicked_route_notice": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_clicked_route_selection():
    st.session_state.clicked_start = None
    st.session_state.clicked_end = None
    st.session_state.clicked_last_signature = None
    st.session_state.clicked_route_results = None
    st.session_state.clicked_route_context = None
    st.session_state.clicked_route_notice = ""


def build_clicked_route_context(start_coordinate, end_coordinate, mode, rainfall, show_route_comparison):
    return {
        "start": format_route_point_signature(start_coordinate),
        "end": format_route_point_signature(end_coordinate),
        "mode": mode,
        "rainfall": rainfall,
        "show_route_comparison": bool(show_route_comparison),
    }


def handle_map_click_selection(last_clicked):
    if not last_clicked:
        return

    lat = last_clicked.get("lat")
    lon = last_clicked.get("lng", last_clicked.get("lon"))
    if lat is None or lon is None:
        return

    coordinate = {"lat": float(lat), "lon": float(lon)}
    signature = format_route_point_signature(coordinate)
    if signature == st.session_state.clicked_last_signature:
        return

    if st.session_state.clicked_start is None:
        st.session_state.clicked_start = coordinate
        st.session_state.clicked_route_notice = "출발지가 저장되었습니다. 도착지를 선택해주세요."
    elif st.session_state.clicked_end is None:
        st.session_state.clicked_end = coordinate
        st.session_state.clicked_route_notice = "도착지가 저장되었습니다. 경로 계산 버튼을 눌러주세요."
    else:
        st.session_state.clicked_route_notice = "출발지와 도착지가 이미 선택되었습니다. 초기화 후 다시 선택해주세요."

    st.session_state.clicked_last_signature = signature
    st.session_state.clicked_route_results = None
    st.session_state.clicked_route_context = None
    st.rerun()


def get_temporary_route_score(rainfall, mode):
    base_scores = {
        "10mm/h": 30,
        "30mm/h": 60,
        "50mm/h": 85,
    }
    base_score = base_scores.get(rainfall, 60)
    if mode == "안전경로":
        return max(base_score - 12, 0)
    return base_score


def get_rainfall_alpha(rainfall):
    return {
        "10mm/h": 1.2,
        "30mm/h": 2.0,
        "50mm/h": 3.0,
    }.get(rainfall, 2.0)


def classify_edge_risk(probability):
    if probability >= 0.95:
        return "통행 불가 후보"
    if probability >= 0.85:
        return "회피 우선"
    if probability >= 0.68:
        return "고위험"
    if probability >= 0.50:
        return "주의"
    return "낮음"


def get_route_risk_points(prediction_df):
    required = {"latitude", "longitude", "ai_risk_probability"}
    if prediction_df.empty or not required.issubset(prediction_df.columns):
        return []
    route_df = prediction_df[["latitude", "longitude", "ai_risk_probability"]].dropna()
    return [
        (float(row.latitude), float(row.longitude), float(row.ai_risk_probability))
        for row in route_df.itertuples(index=False)
    ]


def build_route_risk_lookup(risk_points):
    if not risk_points:
        return {}
    if BallTree is None:
        return {"points": risk_points}
    coordinates = [
        [math.radians(point[0]), math.radians(point[1])]
        for point in risk_points
    ]
    return {
        "tree": BallTree(coordinates, metric="haversine"),
        "risks": [point[2] for point in risk_points],
        "points": risk_points,
    }


def nearest_risk_probability(lat, lon, risk_lookup):
    if not risk_lookup:
        return 0.0
    if risk_lookup.get("tree") is not None:
        _, indexes = risk_lookup["tree"].query([[math.radians(lat), math.radians(lon)]], k=1)
        return risk_lookup["risks"][int(indexes[0][0])]
    risk_points = risk_lookup.get("points", [])
    nearest = min(
        risk_points,
        key=lambda point: (point[0] - lat) ** 2 + (point[1] - lon) ** 2,
    )
    return nearest[2]


def edge_length_m(graph, u, v, data):
    if data and data.get("length") is not None:
        return float(data["length"])
    start = {"lat": float(graph.nodes[u]["y"]), "lon": float(graph.nodes[u]["x"])}
    end = {"lat": float(graph.nodes[v]["y"]), "lon": float(graph.nodes[v]["x"])}
    return calculate_distance_km(start, end) * 1000


def best_edge_data(graph, u, v, weight_key="length"):
    edge_data = graph.get_edge_data(u, v, default={})
    if not edge_data:
        return {}
    if all(isinstance(value, dict) for value in edge_data.values()):
        return min(
            edge_data.values(),
            key=lambda item: float(item.get(weight_key, item.get("length", 0)) or 0),
        )
    return edge_data


def sample_edge_points(graph, u, v, data):
    geometry = data.get("geometry") if data else None
    length = edge_length_m(graph, u, v, data)
    if geometry is not None and hasattr(geometry, "interpolate"):
        sample_count = 3
        if length > 900:
            sample_count = 5
        return [
            (float(point.y), float(point.x))
            for point in (
                geometry.interpolate(index / (sample_count - 1), normalized=True)
                for index in range(sample_count)
            )
        ]

    start_lat = float(graph.nodes[u]["y"])
    start_lon = float(graph.nodes[u]["x"])
    end_lat = float(graph.nodes[v]["y"])
    end_lon = float(graph.nodes[v]["x"])
    return [
        (start_lat, start_lon),
        ((start_lat + end_lat) / 2, (start_lon + end_lon) / 2),
        (end_lat, end_lon),
    ]


def calculate_edge_risk(edge_geometry, ai_grid):
    if not edge_geometry or not ai_grid:
        return {
            "risk": 0.0,
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "risk_class": "낮음",
        }

    risks = [
        nearest_risk_probability(lat, lon, ai_grid)
        for lat, lon in edge_geometry
    ]
    avg_risk = sum(risks) / len(risks)
    max_risk = max(risks)
    edge_risk = (0.6 * max_risk) + (0.4 * avg_risk)
    return {
        "risk": edge_risk,
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "risk_class": classify_edge_risk(edge_risk),
    }


def calculate_safe_edge_cost(length, risk, rainfall):
    alpha = get_rainfall_alpha(rainfall)
    if risk >= 0.95:
        return length * 500
    if risk >= 0.85:
        tier_multiplier = 16.0
    elif risk >= 0.68:
        tier_multiplier = 8.0
    elif risk >= 0.50:
        tier_multiplier = 3.0
    else:
        tier_multiplier = 0.35
    return length * (1 + alpha * risk * tier_multiplier)


def iter_graph_edges(graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
        return
    for u, v, data in graph.edges(data=True):
        yield u, v, None, data


def get_route_corridor_margin(rainfall, multiplier=1.0):
    base_margin = {
        "10mm/h": 0.005,
        "30mm/h": 0.008,
        "50mm/h": 0.011,
    }.get(rainfall, 0.008)
    return base_margin * multiplier


def build_route_corridor_graph(graph, route_nodes, rainfall, multiplier=1.0):
    if not route_nodes:
        return graph
    lats = [float(graph.nodes[node]["y"]) for node in route_nodes]
    lons = [float(graph.nodes[node]["x"]) for node in route_nodes]
    margin = get_route_corridor_margin(rainfall, multiplier)
    min_lat, max_lat = min(lats) - margin, max(lats) + margin
    min_lon, max_lon = min(lons) - margin, max(lons) + margin
    route_node_set = set(route_nodes)
    candidate_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if (
            node in route_node_set
            or (
                min_lat <= float(attrs.get("y", 0)) <= max_lat
                and min_lon <= float(attrs.get("x", 0)) <= max_lon
            )
        )
    ]
    if len(candidate_nodes) < len(route_nodes):
        return graph
    return graph.subgraph(candidate_nodes).copy()


def apply_safe_edge_weights(graph, risk_points, rainfall):
    risk_signature = (
        rainfall,
        len(risk_points),
        round(sum(point[2] for point in risk_points), 4),
    )
    if graph.graph.get("safe_weight_signature") == risk_signature:
        return

    risk_lookup = build_route_risk_lookup(risk_points)
    for u, v, _key, data in iter_graph_edges(graph):
        length = edge_length_m(graph, u, v, data)
        risk_data = calculate_edge_risk(sample_edge_points(graph, u, v, data), risk_lookup)
        data["route_risk_probability"] = risk_data["risk"]
        data["route_avg_risk_probability"] = risk_data["avg_risk"]
        data["route_max_risk_probability"] = risk_data["max_risk"]
        data["route_risk_class"] = risk_data["risk_class"]
        data["route_blocked_candidate"] = risk_data["max_risk"] >= 0.95
        data["safe_length"] = calculate_safe_edge_cost(length, risk_data["risk"], rainfall)

    graph.graph["safe_weight_signature"] = risk_signature


def apply_risk_to_route_edges(graph, route_nodes, risk_points, rainfall):
    if not route_nodes or not risk_points:
        return
    risk_lookup = build_route_risk_lookup(risk_points)
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = best_edge_data(graph, u, v, "length")
        if not data:
            continue
        length = edge_length_m(graph, u, v, data)
        risk_data = calculate_edge_risk(sample_edge_points(graph, u, v, data), risk_lookup)
        data["route_risk_probability"] = risk_data["risk"]
        data["route_avg_risk_probability"] = risk_data["avg_risk"]
        data["route_max_risk_probability"] = risk_data["max_risk"]
        data["route_risk_class"] = risk_data["risk_class"]
        data["route_blocked_candidate"] = risk_data["max_risk"] >= 0.95
        data["safe_length"] = calculate_safe_edge_cost(length, risk_data["risk"], rainfall)


def route_to_coordinates(graph, route_nodes, weight_key="length"):
    coordinates = []
    if len(route_nodes) == 1:
        node = graph.nodes[route_nodes[0]]
        return [[float(node["y"]), float(node["x"])]]

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = best_edge_data(graph, u, v, weight_key)
        geometry = data.get("geometry") if data else None
        if geometry is not None and hasattr(geometry, "coords"):
            segment = [[float(lat), float(lon)] for lon, lat in geometry.coords]
        else:
            segment = [
                [float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])],
                [float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])],
            ]

        if coordinates and segment and coordinates[-1] == segment[0]:
            coordinates.extend(segment[1:])
        else:
            coordinates.extend(segment)

    return coordinates


def collect_route_edges(graph, route_nodes, weight_key):
    route_edges = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = best_edge_data(graph, u, v, weight_key)
        length = edge_length_m(graph, u, v, data)
        risk = float(data.get("route_risk_probability", 0) or 0)
        avg_risk = float(data.get("route_avg_risk_probability", risk) or risk)
        max_risk = float(data.get("route_max_risk_probability", risk) or risk)
        route_edges.append(
            {
                "length_m": length,
                "risk": risk,
                "avg_risk": avg_risk,
                "max_risk": max_risk,
                "risk_class": data.get("route_risk_class", classify_edge_risk(risk)),
                "blocked_candidate": bool(data.get("route_blocked_candidate", False)),
                "cost": float(data.get(weight_key, length) or length),
            }
        )
    return route_edges


def summarize_route(route_edges, rainfall=None, mode="최단경로", fallback_distance_km=0):
    if not route_edges:
        return {
            "distance_km": round(fallback_distance_km, 2),
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "high_risk_count": 0,
            "high_risk_km": 0.0,
            "blocked_count": 0,
            "cost_score": round(fallback_distance_km, 1),
            "risk_score": get_temporary_route_score(rainfall, mode) if rainfall else 0,
        }

    total_m = sum(edge["length_m"] for edge in route_edges)
    total_cost = sum(edge["cost"] for edge in route_edges)
    if total_m <= 0:
        avg_risk = 0.0
    else:
        avg_risk = sum(edge["risk"] * edge["length_m"] for edge in route_edges) / total_m
    max_risk = max(edge["max_risk"] for edge in route_edges)
    high_edges = [edge for edge in route_edges if edge["risk"] >= 0.68 or edge["max_risk"] >= 0.85]
    blocked_edges = [edge for edge in route_edges if edge["blocked_candidate"] or edge["max_risk"] >= 0.95]
    base_score = get_temporary_route_score(rainfall, "최단경로") if rainfall else 50
    risk_score = round(
        min(
            100,
            (base_score * 0.25)
            + (avg_risk * 100 * 0.45)
            + (max_risk * 100 * 0.25)
            + (len(high_edges) * 1.4)
            + (len(blocked_edges) * 8),
        )
    )
    return {
        "distance_km": round(total_m / 1000, 2),
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "high_risk_count": len(high_edges),
        "high_risk_km": round(sum(edge["length_m"] for edge in high_edges) / 1000, 2),
        "blocked_count": len(blocked_edges),
        "cost_score": round(total_cost / 1000, 1),
        "risk_score": risk_score,
    }


def fallback_route_result(start_point, end_point, mode, rainfall, reason):
    distance_km = calculate_distance_km(start_point, end_point)
    summary = summarize_route([], rainfall, mode, distance_km)
    return {
        "coordinates": [[start_point["lat"], start_point["lon"]], [end_point["lat"], end_point["lon"]]],
        "distance_km": distance_km,
        "score": summary["risk_score"],
        "route_type": reason,
        "is_network_route": False,
        "summary": summary,
        "start_point": start_point,
        "end_point": end_point,
    }


def build_route_result(graph, route_nodes, weight_key, mode, rainfall, route_type, start_point, end_point):
    coordinates = route_to_coordinates(graph, route_nodes, weight_key)
    route_edges = collect_route_edges(graph, route_nodes, weight_key)
    summary = summarize_route(route_edges, rainfall, mode)
    return {
        "coordinates": coordinates,
        "distance_km": summary["distance_km"],
        "score": summary["risk_score"],
        "route_type": route_type,
        "is_network_route": True,
        "route_edges": route_edges,
        "summary": summary,
        "start_point": start_point,
        "end_point": end_point,
    }


def build_route_warnings(shortest_route, safe_route):
    warnings = []
    shortest_summary = shortest_route["summary"]
    safe_summary = safe_route["summary"]

    if shortest_summary["max_risk"] >= 0.95:
        warnings.append("최단경로에 매우 위험한 침수 구간이 포함되어 있습니다.")
    elif shortest_summary["max_risk"] >= 0.85:
        warnings.append("최단경로에 고위험 침수 구간이 포함되어 있습니다.")

    if safe_summary["max_risk"] >= 0.95:
        warnings.append("안전경로도 매우 위험한 구간을 완전히 피하지 못했습니다.")
    elif safe_summary["max_risk"] >= 0.85:
        warnings.append("안전경로에 일부 고위험 구간이 남아 있습니다.")

    if (
        shortest_summary["high_risk_count"] > 0
        and safe_summary["high_risk_count"] >= shortest_summary["high_risk_count"]
    ):
        warnings.append("대체 가능한 안전 도로가 부족해 일부 위험 구간이 포함되었습니다.")

    return warnings


def build_route_improvement_text(shortest_route, safe_route):
    shortest_summary = shortest_route["summary"]
    safe_summary = safe_route["summary"]
    distance_delta = safe_summary["distance_km"] - shortest_summary["distance_km"]
    risk_delta = (shortest_summary["avg_risk"] - safe_summary["avg_risk"]) * 100

    if risk_delta > 0.5:
        return (
            f"안전경로는 거리가 {distance_delta:+.2f}km 변하지만, "
            f"평균 위험도를 {risk_delta:.1f}%p 낮춥니다."
        )
    return "두 경로의 평균 위험도 차이가 크지 않습니다."


def calculate_route_results(start_point, end_point, mode, rainfall, ai_predictions, need_safe_route=True):
    straight_shortest = fallback_route_result(start_point, end_point, "최단경로", rainfall, "직선 연결")
    straight_safe = fallback_route_result(start_point, end_point, "안전경로", rainfall, "직선 연결")

    if start_point["name"] == end_point["name"]:
        warning = "출발지와 도착지가 같아 경로 비교를 생략했습니다."
        return {
            "shortest": straight_shortest,
            "safe": straight_safe,
            "selected": straight_safe if mode == "안전경로" else straight_shortest,
            "warnings": [warning],
            "improvement_text": "출발지와 도착지가 같습니다.",
        }

    if calculate_distance_km(start_point, end_point) < CLICK_ROUTE_MIN_DISTANCE_KM:
        warning = "출발지와 도착지가 너무 가까워 경로 비교를 생략했습니다."
        return {
            "shortest": straight_shortest,
            "safe": straight_safe,
            "selected": straight_safe if mode == "안전경로" else straight_shortest,
            "warnings": [warning],
            "improvement_text": "두 지점이 너무 가깝습니다.",
        }

    graph = load_road_graph(build_road_graph_signature())
    if graph is None:
        warning = "도로망 데이터를 불러오지 못해 직선 연결로 표시합니다."
        return {
            "shortest": straight_shortest,
            "safe": straight_safe,
            "selected": straight_safe if mode == "안전경로" else straight_shortest,
            "warnings": [warning],
            "improvement_text": "도로망 데이터가 없어 안전경로 비교를 계산하지 못했습니다.",
        }

    try:
        origin_node, origin_distance_m = nearest_graph_node_with_distance(graph, start_point)
        destination_node, destination_distance_m = nearest_graph_node_with_distance(graph, end_point)
        if max(origin_distance_m, destination_distance_m) > CLICK_ROUTE_MAX_NODE_DISTANCE_M:
            warning = "선택한 지점이 도로망과 너무 멀어 직선 연결로 표시합니다."
            return {
                "shortest": straight_shortest,
                "safe": straight_safe,
                "selected": straight_safe if mode == "안전경로" else straight_shortest,
                "warnings": [warning],
                "improvement_text": "가까운 도로 노드를 안정적으로 찾지 못했습니다.",
            }

        risk_points = get_route_risk_points(ai_predictions)
        warnings = []
        if not risk_points:
            warnings.append("AI 위험 격자가 없어 안전경로는 최단경로 기준으로 표시됩니다.")

        shortest_nodes = nx.shortest_path(graph, origin_node, destination_node, weight="length")
        if risk_points and need_safe_route:
            apply_risk_to_route_edges(graph, shortest_nodes, risk_points, rainfall)
        shortest_route = build_route_result(
            graph,
            shortest_nodes,
            "length",
            "최단경로",
            rainfall,
            "OSM 최단경로",
            start_point,
            end_point,
        )

        if risk_points and need_safe_route:
            try:
                safe_graph = build_route_corridor_graph(graph, shortest_nodes, rainfall)
                apply_safe_edge_weights(safe_graph, risk_points, rainfall)
                safe_nodes = nx.shortest_path(safe_graph, origin_node, destination_node, weight="safe_length")
                safe_route = build_route_result(
                    safe_graph,
                    safe_nodes,
                    "safe_length",
                    "안전경로",
                    rainfall,
                    "AI 위험 회피 경로",
                    start_point,
                    end_point,
                )
            except Exception:
                try:
                    safe_graph = build_route_corridor_graph(graph, shortest_nodes, rainfall, multiplier=1.8)
                    apply_safe_edge_weights(safe_graph, risk_points, rainfall)
                    safe_nodes = nx.shortest_path(safe_graph, origin_node, destination_node, weight="safe_length")
                    safe_route = build_route_result(
                        safe_graph,
                        safe_nodes,
                        "safe_length",
                        "안전경로",
                        rainfall,
                        "AI 위험 회피 경로",
                        start_point,
                        end_point,
                    )
                    warnings.append("일부 구간은 더 넓은 후보 구역에서 우회 경로를 계산했습니다.")
                except Exception:
                    safe_route = build_route_result(
                        graph,
                        shortest_nodes,
                        "length",
                        "안전경로",
                        rainfall,
                        "안전경로 fallback",
                        start_point,
                        end_point,
                    )
                    warnings.append("안전경로 계산에 실패해 최단경로를 대신 표시합니다.")
        else:
            safe_route = build_route_result(
                graph,
                shortest_nodes,
                "length",
                "안전경로",
                rainfall,
                "비교 대기",
                start_point,
                end_point,
            )

        warnings.extend(build_route_warnings(shortest_route, safe_route))
        selected = safe_route if mode == "안전경로" else shortest_route
        return {
            "shortest": shortest_route,
            "safe": safe_route,
            "selected": selected,
            "warnings": warnings,
            "improvement_text": build_route_improvement_text(shortest_route, safe_route),
        }
    except Exception:
        warning = "경로 계산 중 오류가 발생해 직선 연결로 표시합니다."
        return {
            "shortest": straight_shortest,
            "safe": straight_safe,
            "selected": straight_safe if mode == "안전경로" else straight_shortest,
            "warnings": [warning],
            "improvement_text": "경로 계산에 실패했습니다.",
        }


def add_route_markers(map_obj, start_point, end_point):
    folium.Marker(
        location=[start_point["lat"], start_point["lon"]],
        tooltip=start_point.get("tooltip", f"출발지: {start_point['name']}"),
        popup=start_point["name"],
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(map_obj)

    folium.Marker(
        location=[end_point["lat"], end_point["lon"]],
        tooltip=end_point.get("tooltip", f"도착지: {end_point['name']}"),
        popup=end_point["name"],
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(map_obj)


def add_click_route_markers(map_obj, start_point, end_point):
    if start_point is not None:
        folium.Marker(
            location=[start_point["lat"], start_point["lon"]],
            tooltip="클릭 출발지",
            popup="클릭 출발지",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(map_obj)

    if end_point is not None:
        folium.Marker(
            location=[end_point["lat"], end_point["lon"]],
            tooltip="클릭 도착지",
            popup="클릭 도착지",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(map_obj)


def draw_route_line(map_obj, route_result, color, weight, opacity, tooltip, dash_array=None):
    if not route_result or len(route_result.get("coordinates", [])) < 2:
        return
    folium.PolyLine(
        locations=route_result["coordinates"],
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=tooltip,
        dash_array=dash_array,
    ).add_to(map_obj)


def add_comparison_routes_to_map(map_obj, shortest_route, safe_route, selected_mode, show_comparison):
    start_point = shortest_route["start_point"]
    end_point = shortest_route["end_point"]
    add_route_markers(map_obj, start_point, end_point)

    if show_comparison:
        draw_route_line(
            map_obj,
            shortest_route,
            "#2563eb",
            4,
            0.72,
            "최단경로",
            "6",
        )
        draw_route_line(
            map_obj,
            safe_route,
            "#7c3aed",
            6,
            0.9,
            "안전경로",
        )
        return

    selected_route = safe_route if selected_mode == "안전경로" else shortest_route
    draw_route_line(
        map_obj,
        selected_route,
        "#7c3aed" if selected_mode == "안전경로" else "#2563eb",
        6 if selected_mode == "안전경로" else 5,
        0.88,
        selected_mode,
    )


def route_score_color(score):
    if score >= 75:
        return "#dc2626"
    if score >= 50:
        return "#d97706"
    return "#16a34a"


def risk_percent_text(value):
    return f"{value * 100:.1f}%"


def render_route_comparison(shortest_summary, safe_summary, warnings, improvement_text):
    avoided_high_count = max(0, shortest_summary["high_risk_count"] - safe_summary["high_risk_count"])
    warning_html = "".join(f"<div class='route-warning'>{message}</div>" for message in warnings[:3])
    if not warning_html:
        warning_html = "<div class='route-ok'>고위험 경고 없음</div>"

    def route_column(title, summary, accent_color, avoided_text="-"):
        score_color = route_score_color(summary["risk_score"])
        return f"""<div class="route-column">
<div class="route-column-title" style="color:{accent_color};">{title}</div>
<div><span>거리</span><b>{summary['distance_km']:.2f} km</b></div>
<div><span>위험 점수</span><b style="color:{score_color};">{summary['risk_score']}점</b></div>
<div><span>평균 위험도</span><b>{risk_percent_text(summary['avg_risk'])}</b></div>
<div><span>최대 위험도</span><b>{risk_percent_text(summary['max_risk'])}</b></div>
<div><span>고위험 구간</span><b>{summary['high_risk_count']}개 · {summary['high_risk_km']:.2f} km</b></div>
<div><span>회피 구간</span><b>{avoided_text}</b></div>
<div><span>비용 점수</span><b>{summary['cost_score']:.1f}</b></div>
</div>"""

    shortest_column = route_column("최단경로", shortest_summary, "#2563eb")
    safe_column = route_column("안전경로", safe_summary, "#7c3aed", f"{avoided_high_count}개")

    st.markdown(
        f"""
<div class="route-card">
    <div class="section-head route-head">
        <h3>Route Compare</h3>
        <span>AI risk weighted</span>
    </div>
    <div class="route-explain">
        안전경로는 AI 침수 위험 확률을 도로 비용에 반영해 고위험 구간을 우회하도록 계산됩니다.
    </div>
    <div class="route-compare-grid">
{shortest_column}
{safe_column}
    </div>
    <div class="route-summary-line">{improvement_text}</div>
    <div class="route-warning-list">{warning_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_route_selected_card(route_result, mode, rainfall):
    summary = route_result["summary"]
    score_color = route_score_color(summary["risk_score"])
    st.markdown(
        f"""
<div class="route-card">
    <div class="section-head route-head">
        <h3>Route Preview</h3>
        <span>{route_result['route_type']}</span>
    </div>
    <div class="route-grid">
        <div><span>기준</span><b>{mode}</b></div>
        <div><span>거리</span><b>{summary['distance_km']:.2f} km</b></div>
        <div><span>위험 점수</span><b style="color:{score_color}">{summary['risk_score']}점</b></div>
        <div><span>강수량</span><b>{rainfall}</b></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


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

    .route-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 18px;
        margin-top: 14px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .route-head {
        margin-bottom: 14px;
    }

    .route-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 12px;
    }

    .route-grid div {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        background: #f8fafc;
        min-height: 72px;
    }

    .route-grid span {
        display: block;
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .route-grid b {
        display: block;
        color: #172033;
        font-size: 18px;
        line-height: 1.2;
        font-weight: 950;
    }

    .route-explain {
        margin-bottom: 12px;
        color: #475569;
        font-size: 13px;
        line-height: 1.5;
    }

    .route-compare-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }

    .route-column {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        background: #f8fafc;
    }

    .route-column-title {
        grid-column: 1 / -1;
        font-size: 15px;
        font-weight: 950;
    }

    .route-column span {
        display: block;
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .route-column b {
        display: block;
        color: #172033;
        font-size: 15px;
        line-height: 1.2;
        font-weight: 950;
    }

    .route-summary-line {
        margin-top: 12px;
        padding: 10px 12px;
        border-radius: 8px;
        background: #eef6ff;
        color: #075985;
        font-size: 13px;
        font-weight: 850;
    }

    .route-warning-list {
        display: grid;
        gap: 6px;
        margin-top: 10px;
    }

    .route-warning,
    .route-ok {
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 850;
    }

    .route-warning {
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }

    .route-ok {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #bbf7d0;
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

        .route-grid {
            grid-template-columns: 1fr 1fr;
        }

        .route-compare-grid,
        .route-column {
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

initialize_click_route_state()

rainfall = st.sidebar.selectbox(
    "강수량 시나리오 선택",
    ["10mm/h", "30mm/h", "50mm/h"],
)

mode = st.sidebar.radio(
    "경로 추천 기준",
    ["최단경로", "안전경로"],
)
show_route_comparison = st.sidebar.checkbox("두 경로 비교 표시", value=False)

route_points = get_route_points()
route_point_names = list(route_points)

st.sidebar.markdown(
    """
<div style="margin-top:18px; margin-bottom:6px; color:#172033; font-size:14px; font-weight:900;">
    경로 입력 방식
</div>
""",
    unsafe_allow_html=True,
)

route_input_method = st.sidebar.radio(
    "경로 입력 방식",
    ["주요 지점 선택", "지도에서 직접 선택"],
    label_visibility="collapsed",
)

start_point = None
end_point = None
start_name = "미선택"
end_name = "미선택"
click_route_calculate = False
clicked_route_context = None
clicked_start_point = make_clicked_route_point("클릭 출발지", st.session_state.clicked_start)
clicked_end_point = make_clicked_route_point("클릭 도착지", st.session_state.clicked_end)

if route_input_method == "주요 지점 선택":
    st.sidebar.markdown(
        """
<div style="margin-top:18px; margin-bottom:6px; color:#172033; font-size:14px; font-weight:900;">
    경로 지점 선택
</div>
""",
        unsafe_allow_html=True,
    )
    default_start_index = route_point_names.index("강남역") if "강남역" in route_point_names else 0
    default_end_index = route_point_names.index("삼성역") if "삼성역" in route_point_names else 0
    start_name = st.sidebar.selectbox("출발지", route_point_names, index=default_start_index)
    end_name = st.sidebar.selectbox("도착지", route_point_names, index=default_end_index)
    start_point = route_points[start_name]
    end_point = route_points[end_name]
else:
    st.sidebar.markdown(
        """
<div style="margin-top:10px; color:#475569; font-size:13px; line-height:1.6;">
지도에서 첫 번째 클릭은 출발지, 두 번째 클릭은 도착지로 저장됩니다. 두 지점을 선택한 뒤 경로 계산 버튼을 눌러주세요.
</div>
""",
        unsafe_allow_html=True,
    )
    col_reset, col_status = st.sidebar.columns([1, 1])
    with col_reset:
        if st.button(
            "선택 초기화",
            disabled=st.session_state.clicked_start is None and st.session_state.clicked_end is None,
            use_container_width=True,
        ):
            clear_clicked_route_selection()
            st.rerun()
    with col_status:
        st.caption("클릭 선택")

    if clicked_start_point is not None:
        st.sidebar.caption(
            f"출발지: {clicked_start_point['lat']:.5f}, {clicked_start_point['lon']:.5f}"
        )
    else:
        st.sidebar.caption("출발지: 지도에서 선택")

    if clicked_end_point is not None:
        st.sidebar.caption(
            f"도착지: {clicked_end_point['lat']:.5f}, {clicked_end_point['lon']:.5f}"
        )
    else:
        st.sidebar.caption("도착지: 지도에서 선택")

    if st.session_state.clicked_route_notice:
        st.sidebar.info(st.session_state.clicked_route_notice)

    start_point = clicked_start_point
    end_point = clicked_end_point
    start_name = format_route_point_signature(st.session_state.clicked_start)
    end_name = format_route_point_signature(st.session_state.clicked_end)
    clicked_route_context = build_clicked_route_context(
        st.session_state.clicked_start,
        st.session_state.clicked_end,
        mode,
        rainfall,
        show_route_comparison,
    )
    click_route_ready = clicked_start_point is not None and clicked_end_point is not None
    click_route_calculate = st.sidebar.button(
        "경로 계산",
        disabled=not click_route_ready,
        use_container_width=True,
    )
    if (
        click_route_ready
        and st.session_state.clicked_route_results is not None
        and st.session_state.clicked_route_context != clicked_route_context
    ):
        st.sidebar.warning("선택 또는 옵션이 바뀌었습니다. 경로 계산을 다시 눌러주세요.")

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
route_should_calculate_now = route_input_method == "주요 지점 선택" or click_route_calculate
route_needs_ai = route_should_calculate_now and (mode == "안전경로" or show_route_comparison)
if show_ai_layer or route_needs_ai:
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

route_results = None
if route_input_method == "주요 지점 선택":
    route_results = calculate_route_results(
        start_point,
        end_point,
        mode,
        rainfall,
        ai_predictions,
        need_safe_route=route_needs_ai,
    )
elif click_route_calculate and start_point is not None and end_point is not None:
    route_results = calculate_route_results(
        start_point,
        end_point,
        mode,
        rainfall,
        ai_predictions,
        need_safe_route=route_needs_ai,
    )
    st.session_state.clicked_route_results = route_results
    st.session_state.clicked_route_context = clicked_route_context
elif (
    clicked_route_context is not None
    and st.session_state.clicked_route_context == clicked_route_context
    and st.session_state.clicked_route_results is not None
):
    route_results = st.session_state.clicked_route_results

route_result = route_results["selected"] if route_results is not None else None

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
    dragging=route_input_method != "지도에서 직접 선택",
    scrollWheelZoom=route_input_method != "지도에서 직접 선택",
    doubleClickZoom=route_input_method != "지도에서 직접 선택",
    touchZoom=route_input_method != "지도에서 직접 선택",
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

if route_input_method == "지도에서 직접 선택":
    click_select_css = """
    <style>
        .leaflet-container,
        .leaflet-grab,
        .leaflet-dragging .leaflet-grab {
            cursor: crosshair !important;
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(click_select_css))

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
if route_results is not None:
    add_comparison_routes_to_map(
        m,
        route_results["shortest"],
        route_results["safe"],
        mode,
        show_route_comparison,
    )
elif route_input_method == "지도에서 직접 선택":
    add_click_route_markers(m, clicked_start_point, clicked_end_point)

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

map_style_key = "cartodb" if base_map_style == "CartoDB positron" else "osm"
rainfall_key = rainfall.replace("/", "_").replace(" ", "")
route_input_key = "click" if route_input_method == "지도에서 직접 선택" else "preset"
mode_key = "safe" if mode == "안전경로" else "shortest"
if route_input_method == "지도에서 직접 선택":
    start_key = format_route_point_signature(st.session_state.clicked_start).replace(",", "_").replace(".", "p")
    end_key = format_route_point_signature(st.session_state.clicked_end).replace(",", "_").replace(".", "p")
else:
    start_key = str(route_point_names.index(start_name)) if start_name in route_point_names else "0"
    end_key = str(route_point_names.index(end_name)) if end_name in route_point_names else "0"

map_key = (
    f"rainguard_map_v4_{map_style_key}_{rainfall_key}_{int(show_expected)}_"
    f"{int(show_history_2022)}_{int(show_history_2023)}_{int(show_history_other)}_"
    f"{int(show_ai_layer)}_{max_expected_stage}_{mode_key}_{route_input_key}_"
    f"{start_key}_{end_key}_{int(show_route_comparison)}"
)
map_returned_objects = (
    ["last_clicked", "last_object_clicked"]
    if route_input_method == "지도에서 직접 선택"
    else []
)
map_data = st_folium(
    m,
    width=1240,
    height=620,
    key=map_key,
    returned_objects=map_returned_objects,
)

if route_input_method == "지도에서 직접 선택":
    clicked_location = None
    if isinstance(map_data, dict):
        clicked_location = map_data.get("last_clicked") or map_data.get("last_object_clicked")
    handle_map_click_selection(clicked_location)

st.markdown("</div>", unsafe_allow_html=True)

if route_results is not None:
    if show_route_comparison or mode == "안전경로":
        render_route_comparison(
            route_results["shortest"]["summary"],
            route_results["safe"]["summary"],
            route_results["warnings"],
            route_results["improvement_text"],
        )
    else:
        render_route_selected_card(route_result, mode, rainfall)
elif route_input_method == "지도에서 직접 선택":
    st.info("지도에서 출발지와 도착지를 선택한 뒤 경로 계산 버튼을 눌러주세요.")
