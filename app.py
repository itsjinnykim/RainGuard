import streamlit as st
import folium
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


st.set_page_config(
    page_title="RainGuard",
    page_icon="🌧️",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FLOOD_EXPECTED_DIR = DATA_DIR / "flood_expected"
FLOOD_HISTORY_DIR = DATA_DIR / "flood_history"

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
    1: {"color": "#0ea5e9", "fill": "#bae6fd", "opacity": 0.16},
    2: {"color": "#06b6d4", "fill": "#67e8f9", "opacity": 0.16},
    3: {"color": "#14b8a6", "fill": "#5eead4", "opacity": 0.15},
    4: {"color": "#84cc16", "fill": "#bef264", "opacity": 0.15},
    5: {"color": "#f59e0b", "fill": "#fbbf24", "opacity": 0.17},
    6: {"color": "#ef4444", "fill": "#fb7185", "opacity": 0.18},
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


def simplify_for_map(gdf):
    if gdf is None or gdf.empty:
        return None

    mapped = gdf.copy()
    tolerance = 0.00003 if len(mapped) > 1000 else 0.00001
    mapped["geometry"] = mapped.geometry.simplify(tolerance, preserve_topology=True)
    mapped = mapped[mapped.geometry.notna()]
    mapped = mapped[~mapped.geometry.is_empty]
    return mapped


def build_data_signature():
    shp_files = find_shp_files(FLOOD_EXPECTED_DIR) + find_shp_files(FLOOD_HISTORY_DIR)
    return tuple((str(path), path.stat().st_mtime, path.stat().st_size) for path in shp_files)


@st.cache_data(show_spinner=False)
def load_spatial_layers(signature):
    if gpd is None:
        return {
            "expected_by_stage": {},
            "history_2022": None,
            "history_2023": None,
            "summary": [
                {"name": "침수예상도", "files": 0, "features": 0},
                {"name": "2022 침수흔적도", "files": 0, "features": 0},
                {"name": "2023 침수흔적도", "files": 0, "features": 0},
            ],
            "messages": ["GeoPandas가 설치되어 있지 않아 SHP 파일을 읽을 수 없음"],
        }

    expected_files = find_shp_files(FLOOD_EXPECTED_DIR)
    history_files = find_shp_files(FLOOD_HISTORY_DIR)
    history_2022_files = [path for path in history_files if "2022" in path.name]
    history_2023_files = [path for path in history_files if "2023" in path.name]

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

    messages = []
    if not expected_files:
        messages.append("data/flood_expected 폴더에서 .shp 파일을 찾지 못함")
    if not history_2022_files:
        messages.append("data/flood_history 폴더에서 2022 .shp 파일을 찾지 못함")
    if not history_2023_files:
        messages.append("data/flood_history 폴더에서 2023 .shp 파일을 찾지 못함")

    messages.extend(expected_messages)
    messages.extend(history_2022_messages)
    messages.extend(history_2023_messages)

    return {
        "expected_by_stage": expected_by_stage,
        "expected_stage_counts": expected_stage_counts,
        "history_2022": simplify_for_map(history_2022),
        "history_2023": simplify_for_map(history_2023),
        "summary": [
            {"name": "침수예상도", "files": len(expected_files), "features": expected_count},
            {"name": "2022 침수흔적도", "files": len(history_2022_files), "features": history_2022_count},
            {"name": "2023 침수흔적도", "files": len(history_2023_files), "features": history_2023_count},
        ],
        "messages": messages,
    }


def add_polygon_layer(map_obj, gdf, name, color, fill_color, fill_opacity, weight, show, dash_array=None):
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
show_history_2022 = st.sidebar.checkbox("2022 침수흔적도 표시", value=True)
show_history_2023 = st.sidebar.checkbox("2023 침수흔적도 표시", value=False)

risk = RISK_BY_RAINFALL[rainfall]

with st.spinner("SHP 공간 데이터를 불러오는 중입니다..."):
    spatial_layers = load_spatial_layers(build_data_signature())

max_expected_stage = EXPECTED_STAGE_BY_RAINFALL[rainfall]
visible_expected_stages = [
    stage
    for stage in sorted(spatial_layers["expected_by_stage"])
    if stage <= max_expected_stage
]
visible_expected_count = sum(
    spatial_layers["expected_stage_counts"].get(stage, 0)
    for stage in visible_expected_stages
)
expected_stage_label = f"1~{max_expected_stage}단계"

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
        <div class="metric-note">SHP 데이터 기반 시각화</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">침수예상도 단계</div>
        <div class="metric-value">{expected_stage_label}</div>
        <div class="metric-note">강수량 시나리오 적용</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">강수량 시나리오</div>
        <div class="metric-value">{rainfall}</div>
        <div class="metric-note">기상청 API 연동 예정</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">침수 위험도</div>
        <div class="metric-value" style="color:{risk['color']}">{risk['level']}</div>
        <div class="metric-note">{risk['score']}/100 · {risk['summary']}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


analysis_center = [37.5172, 127.0473]

m = folium.Map(
    location=analysis_center,
    zoom_start=13,
    tiles="OpenStreetMap",
)

for stage in visible_expected_stages:
    style = EXPECTED_STAGE_STYLE.get(stage, EXPECTED_STAGE_STYLE[6])
    add_polygon_layer(
        m,
        spatial_layers["expected_by_stage"].get(stage),
        f"침수예상도 {stage}단계",
        color=style["color"],
        fill_color=style["fill"],
        fill_opacity=style["opacity"],
        weight=1.0,
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

folium.Marker(
    location=analysis_center,
    popup="RainGuard 분석 중심",
    tooltip="분석 중심",
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

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
    <div style="font-weight: 900; margin-bottom: 6px;">SHP 지도 레이어</div>
    <div><span style="display:inline-block;width:10px;height:10px;background:#38bdf8;border:1px solid #0284c7;margin-right:6px;"></span>침수예상도 {expected_stage_label}</div>
    <div><span style="display:inline-block;width:10px;height:10px;background:#facc15;border:1px solid #d97706;margin-right:6px;"></span>2022 침수흔적도</div>
    <div><span style="display:inline-block;width:10px;height:10px;background:#fb7185;border:1px solid #ef4444;margin-right:6px;"></span>2023 침수흔적도</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st.markdown(
    f"""
<div class="map-card">
    <div class="section-head">
        <h3>침수예상도 / 침수흔적도 지도</h3>
        <span>{rainfall} · {expected_stage_label} · {mode}</span>
    </div>
    <div class="source-note">
        데이터: <b>data/flood_expected</b>, <b>data/flood_history</b> 폴더의 압축 해제된 SHP 파일 · 2022/2023 침수흔적도는 우측 레이어 버튼에서 켤 수 있습니다.
    </div>
""",
    unsafe_allow_html=True,
)

map_key = (
    f"rainguard_map_{rainfall}_{show_expected}_"
    f"{show_history_2022}_{show_history_2023}_{expected_stage_label}"
)
st_folium(m, width=1240, height=620, key=map_key)

st.markdown("</div>", unsafe_allow_html=True)
