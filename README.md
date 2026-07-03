# RainGuard

공공데이터 기반 도시 침수 위험 예측 및 안전 이동 경로 추천 서비스 MVP입니다.

현재 버전은 제안서 제출 전 구현 가능성을 보여주기 위한 Streamlit 프로토타입입니다. 지도에는 서울안전누리 안전정보지도 > 태풍·호우 > 침수흔적도 공식 WMS 레이어를 표시합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 현재 구현

- 강수량 시나리오 선택: `10mm/h`, `30mm/h`, `50mm/h`
- 서울안전누리 침수흔적도 연도 선택
- 시나리오별 침수 위험도 카드 표시
- Folium 기반 지도 시각화
- 공식 침수흔적도 WMS 이미지 오버레이 표시

## 확정 데이터 출처

- 서울안전누리 침수흔적도: https://safecity.seoul.go.kr/distFclt/cfMapDs/cfMapDs.page?menuId=MENU_SSNS_000014
- 기상청 단기예보 조회서비스: https://www.data.go.kr/data/15084084/openapi.do

## 다음 고도화

- 기상청 단기예보 조회서비스 API 연동
- 격자 단위 학습 데이터 생성
- RandomForest 기반 침수 위험도 예측
- 최단경로와 안전경로 비교
