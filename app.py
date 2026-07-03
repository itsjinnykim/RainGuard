import streamlit as st
import folium
from urllib.parse import quote
from streamlit_folium import st_folium


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

FLOOD_TRACE_YEARS = [
    "2025",
    "2024",
    "2023",
    "2022",
    "2020",
    "2019",
    "2018",
    "2017",
    "2016",
    "2014",
    "2013",
    "2012",
    "2011",
    "2010",
]

SAFECITY_MAP_URL = "https://safecity.seoul.go.kr/distFclt/cfMapDs/cfMapDs.page?menuId=MENU_SSNS_000014"
SAFECITY_WMS_URL = "https://safecity.seoul.go.kr/G2DataService/GService"
FLOOD_TRACE_BBOX_5186 = "196903.25,539510.30,211944.60,551726.89"
FLOOD_TRACE_BOUNDS_WGS84 = [
    [37.455, 126.965],
    [37.565, 127.135],
]

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


def build_flood_trace_wms_url(year):
    layer_name = quote(f"수방 침수흔적도 {year}", safe="")
    return (
        f"{SAFECITY_WMS_URL}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&FORMAT=image/png&TRANSPARENT=TRUE&LAYERS={layer_name}"
        f"&CRS=EPSG:5186&BBOX={FLOOD_TRACE_BBOX_5186}&WIDTH=1100&HEIGHT=800"
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

flood_year = st.sidebar.selectbox(
    "침수흔적도 연도",
    FLOOD_TRACE_YEARS,
    index=FLOOD_TRACE_YEARS.index("2022"),
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
    침수흔적도: <b>{flood_year}년</b><br>
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


st.markdown(
    f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">분석 지역</div>
        <div class="metric-value">강남구</div>
        <div class="metric-note">서울안전누리 공식 레이어</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">침수흔적도</div>
        <div class="metric-value">{flood_year}</div>
        <div class="metric-note">태풍·호우 > 침수흔적도</div>
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

folium.raster_layers.ImageOverlay(
    name=f"서울안전누리 침수흔적도 {flood_year}",
    image=build_flood_trace_wms_url(flood_year),
    bounds=FLOOD_TRACE_BOUNDS_WGS84,
    opacity=0.62,
    interactive=False,
    cross_origin=False,
    zindex=2,
).add_to(m)

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
    <div style="font-weight: 900; margin-bottom: 4px;">공식 침수흔적도</div>
    <div><span style="display:inline-block;width:10px;height:10px;background:#f5e84a;border:1px solid #d4c900;margin-right:6px;"></span>{flood_year}년 침수흔적</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st.markdown(
    f"""
<div class="map-card">
    <div class="section-head">
        <h3>침수흔적도 지도</h3>
        <span>{flood_year}년 · {rainfall} · {mode}</span>
    </div>
    <div class="source-note">
        데이터 출처: <a href="{SAFECITY_MAP_URL}" target="_blank">서울안전누리 안전정보지도 > 태풍·호우 > 침수흔적도</a>
    </div>
""",
    unsafe_allow_html=True,
)

st_folium(m, width=1240, height=620)

st.markdown("</div>", unsafe_allow_html=True)
