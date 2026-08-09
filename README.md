# RainGuard

RainGuard는 공공데이터 기반 도시 침수 위험 예측 및 안전 이동 경로 추천 Streamlit 웹앱입니다. 현재 구현 범위는 강남구 프로토타입이며, 강수량 시나리오, 침수예상도, 침수흔적도, AI 위험 확률, OSM 도로망을 함께 사용해 폭우 상황에서 최단경로와 안전경로를 비교합니다.

## 구현 완료 범위

### Streamlit 웹앱

- 강남구 침수 위험 지도와 경로 추천 화면 구현
- 강수량 시나리오 선택: 10mm/h, 30mm/h, 50mm/h
- 경로 추천 기준 선택: 최단경로, 안전경로
- 최단경로와 안전경로 동시 비교 표시 옵션
- 지도 스타일 선택: CartoDB positron, OpenStreetMap
- 침수예상도, 침수흔적도, AI 예측 격자 레이어 표시 옵션
- 불러온 SHP 파일 수, polygon 수, 로딩 메시지를 사이드바 상태 카드로 표시
- AI 평균 위험확률, 위험 격자 수, Recall 등 주요 지표 요약 카드 표시

### 지도 기반 경로 입력

- 강남구 주요 지점 selectbox 기반 출발지/도착지 선택
- 지도에서 직접 출발지와 도착지를 클릭해 선택하는 기능
- 클릭 선택 초기화, 선택 상태 표시, 경로 계산 버튼 구현
- 옵션 또는 선택 지점이 바뀐 경우 재계산 안내 표시
- 지도 클릭 선택용 경량 지도와 경로 결과 지도를 분리해 렌더링 성능 개선

### 공간 데이터 시각화

- `data/flood_expected`의 서울시 침수예상도 SHP 파일 로딩 및 단계별 표시
- 강수량 시나리오에 따라 표시 침수심 단계 자동 조정
  - 10mm/h: 1~2단계
  - 30mm/h: 1~4단계
  - 50mm/h: 1~6단계
- `data/flood_history`의 침수흔적도 SHP 파일 연도 자동 탐색
- 기본 침수흔적도 표시 연도: 2022년, 2023년 데이터가 있을 때 자동 선택
- SHP 좌표계 추정, EPSG:4326 변환, 강남구 분석 범위 클리핑, 지도 표시용 geometry 단순화
- SHP sidecar 누락, CRS 누락, geometry 누락 등의 문제를 사용자 메시지로 표시

### AI 침수 위험 예측

- RandomForest 기반 격자별 침수 위험 확률 예측
- `data/processed/flood_dataset.csv`와 `models/flood_random_forest.joblib` 로딩
- `data/processed/flood_grid.geojson`이 있으면 실제 격자 geometry로 AI 위험 격자 표시
- AI 위험 확률 50% 이상인 격자만 지도에 표시
- 위험 등급 구분
  - 50% 이상: 주의
  - 68% 이상: 높음
  - 85% 이상: 매우 높음
- 모델 feature
  - `latitude`, `longitude`
  - `rainfall_mm_h`
  - `expected_stage`
  - `history_polygon_count`
  - `history_recent_polygon_count`
  - `history_year_count`
  - `years_since_latest_history`
  - `history_recency_score`
  - `distance_to_flood_area_m`

### 도로망 기반 경로 추천

- OSMnx 기반 강남구 보행 도로망 사용
- `data/road_network/gangnam_walk.pkl` 우선 로딩, 없으면 `gangnam_walk.graphml` 로딩
- 최단경로는 도로 길이 `length` 기준으로 계산
- 안전경로는 도로 구간별 AI 위험도를 반영한 `safe_length` 기준으로 계산
- 경로 구간 risk는 도로 geometry 샘플 지점의 AI 위험 확률을 이용해 계산

```text
edge_risk = 0.6 x max_risk + 0.4 x avg_risk
safe_length = length x (1 + alpha x edge_risk x tier_multiplier)
```

강수량별 alpha 값은 다음과 같습니다.

| 강수량 | alpha |
| --- | ---: |
| 10mm/h | 1.2 |
| 30mm/h | 2.0 |
| 50mm/h | 3.0 |

위험도 구간별 가중치는 다음과 같습니다.

| edge_risk | 처리 방식 |
| --- | --- |
| 0.95 이상 | `length x 500`으로 사실상 회피 후보 처리 |
| 0.85 이상 | `tier_multiplier = 16.0` |
| 0.68 이상 | `tier_multiplier = 8.0` |
| 0.50 이상 | `tier_multiplier = 3.0` |
| 0.50 미만 | `tier_multiplier = 0.35` |

안전경로 탐색은 최단경로 주변 corridor graph에서 먼저 수행하고, 실패하면 더 넓은 후보 구역에서 재시도합니다. 도로망 데이터가 없거나 클릭 지점이 도로망에서 너무 멀면 직선 연결로 fallback합니다.

### 경로 비교 결과

- 선택 경로 지도 표시
- 최단경로와 안전경로 동시 지도 표시
- 경로별 거리
- 위험 점수
- 평균 위험도
- 최대 위험도
- 고위험 구간 개수와 길이
- 통행 불가 후보 구간 개수
- 비용 점수
- 안전경로가 줄인 평균 위험도와 거리 변화 요약
- 고위험 구간 포함 여부에 대한 경고 메시지

## 모델 검증 결과

| 검증 방식 | Accuracy | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| 랜덤 분할 | 0.9673 | 1.0000 | 0.9533 | 0.9808 |
| 동서 4-band 공간 검증 평균 | 0.9587 | 0.9596 | 0.9515 | 0.9838 |

동쪽 지역 holdout 검증에서는 모든 지표가 1.0000으로 나왔지만, 특정 영역이 비교적 쉬운 테스트 영역일 수 있으므로 대표 성능으로 과도하게 강조하지 않습니다. 자세한 검증 설명은 [`docs/proposal_model_validation.md`](docs/proposal_model_validation.md)를 참고하세요.

### 주요 feature importance

| Feature | Importance |
| --- | ---: |
| `expected_stage` | 0.280766 |
| `rainfall_mm_h` | 0.193947 |
| `distance_to_flood_area_m` | 0.179272 |
| `history_recent_polygon_count` | 0.070751 |
| `history_polygon_count` | 0.069633 |

## 활용 데이터와 생성 산출물

- 서울안전누리 안전정보지도 침수예상도 SHP
- 서울시 침수흔적도 SHP
- 강남구 경계 GeoJSON
- 강남구 500m 격자 기반 학습 데이터셋
- 강수량 시나리오별 AI 예측 데이터
- OSMnx 강남구 보행 도로망 GraphML 및 pickle

주요 생성 파일은 다음 경로를 사용합니다.

```text
data/processed/flood_dataset.csv
data/processed/flood_grid.geojson
data/road_network/gangnam_walk.graphml
data/road_network/gangnam_walk.pkl
models/flood_random_forest.joblib
models/model_metrics.csv
models/model_metrics_random.csv
models/model_metrics_spatial.csv
models/model_metrics_spatial_band_summary.csv
models/feature_importance.csv
```

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

공간 데이터, 모델 파일, 도로망 파일이 준비되어 있으면 전체 기능이 동작합니다. 일부 데이터가 없을 때는 앱이 가능한 범위에서 레이어를 건너뛰거나 직선 경로 fallback으로 표시합니다.

## 데이터 생성 스크립트

### 침수 학습 데이터셋 생성

```bash
python scripts/build_flood_dataset.py
```

`data/flood_expected`, `data/flood_history`, `data/boundary/gangnam_boundary.geojson`을 기반으로 강남구 격자 학습 데이터셋과 격자 GeoJSON을 생성합니다.

### 도로망 생성

```bash
python scripts/build_road_network.py
```

강남구 경계 GeoJSON을 기반으로 OSMnx 보행 도로망을 내려받아 GraphML과 빠른 로딩용 pickle 파일로 저장합니다.

## 현재 한계와 다음 단계

- 현재 분석 범위는 강남구 프로토타입입니다.
- 강수량은 실시간 예보 API가 아니라 10mm/h, 30mm/h, 50mm/h 시나리오입니다.
- 모델 검증 결과는 현재 보유한 강남구 프로토타입 데이터셋 기준입니다.
- 향후 기상청 단기예보조회서비스 API 연동, 더 긴 기간의 침수흔적도 추가, 배수시설/지형/도로 고도 데이터 반영, 다른 자치구 확장을 진행할 예정입니다.

https://itsjinnykim-rainguard-app-sdxonp.streamlit.app/
