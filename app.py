import streamlit as st
import folium
import csv
from pathlib import Path
from streamlit_folium import st_folium


BASE_DIR = Path(__file__).resolve().parent
RISK_POINTS_PATH = BASE_DIR / "data" / "sample_risk_points.csv"


st.set_page_config(
    page_title="RainGuard",
    page_icon="🌧️",
    layout="wide",
)


RISK_BY_RAINFALL = {
    "10mm/h": {
        "level": "낮음",
        "score": 32,
        "color": "#22c55e",
        "circle": "green",
        "summary": "일부 저지대 주의",
    },
    "30mm/h": {
        "level": "주의",
        "score": 64,
        "color": "#f59e0b",
        "circle": "orange",
        "summary": "침수 취약지 위험 상승",
    },
    "50mm/h": {
        "level": "높음",
        "score": 87,
        "color": "#ef4444",
        "circle": "red",
        "summary": "우회 경로 권장",
    },
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


def load_risk_points():
    if not RISK_POINTS_PATH.exists():
        return [
            {"latitude": 37.5172, "longitude": 127.0473, "name": "구청 인근"},
            {"latitude": 37.4981, "longitude": 127.0276, "name": "주요 역세권"},
            {"latitude": 37.5045, "longitude": 127.0490, "name": "업무지구 인근"},
            {"latitude": 37.5140, "longitude": 127.0600, "name": "간선도로 인근"},
            {"latitude": 37.4897, "longitude": 127.0660, "name": "저지대 주거지 인근"},
        ]

    with RISK_POINTS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "name": row["name"],
            }
            for row in csv.DictReader(file)
        ]


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

risk = RISK_BY_RAINFALL[rainfall]

st.sidebar.markdown(
    f"""
<div style="margin-top:22px; color:#334155; font-size:15px; line-height:1.7;">
    선택한 강수량: <b>{rainfall}</b><br>
    위험도: <b style="color:{risk['color']}">{risk['level']}</b>
</div>
""",
    unsafe_allow_html=True,
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


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
<div class="metric-card">
    <div class="metric-label">분석 지역</div>
    <div class="metric-value">강남구</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">강수량</div>
    <div class="metric-value">{rainfall}</div>
    <div class="metric-note">시나리오 기반 시연</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">침수 위험도</div>
    <div class="metric-value" style="color:{risk['color']}">{risk['level']}</div>
    <div class="metric-note">{risk['score']}/100 · {risk['summary']}</div>
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

risk_points = load_risk_points()

for point in risk_points:
    lat = point["latitude"]
    lon = point["longitude"]
    name = point["name"]

    folium.Circle(
        location=[lat, lon],
        radius=360,
        color=risk["circle"],
        fill=True,
        fill_color=risk["circle"],
        fill_opacity=0.28,
        weight=2,
        tooltip=f"{name} · {risk['level']} 위험",
        popup=f"""
        <b>{name}</b><br>
        강수량: {rainfall}<br>
        침수 위험도: {risk['level']}<br>
        위험 점수: {risk['score']}/100
        """,
    ).add_to(m)

folium.Marker(
    location=analysis_center,
    popup="RainGuard 분석 중심",
    tooltip="분석 중심",
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(m)

st.markdown(
    f"""
<div class="map-card">
    <div class="section-head">
    <h3>침수 위험 지도</h3>
        <span>{rainfall} · {mode} · 위험도 {risk['score']}/100</span>
    </div>
""",
    unsafe_allow_html=True,
)

st_folium(m, width=1240, height=620)

st.markdown("</div>", unsafe_allow_html=True)
