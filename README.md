# RainGuard

공공데이터 기반 도시 침수 위험 예측 및 안전 이동 경로 추천 서비스 MVP입니다.

현재 버전은 제안서 제출 전 구현 가능성을 보여주기 위한 Streamlit 프로토타입입니다. 지도에는 `data/flood_expected`, `data/flood_history` 폴더의 압축 해제된 SHP 파일을 표시합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 현재 구현

- 강수량 시나리오 선택: `10mm/h`, `30mm/h`, `50mm/h`
- 침수예상도, 2022 침수흔적도, 2023 침수흔적도 SHP 자동 로딩
- 시나리오별 침수 위험도 카드 표시
- Folium 기반 지도 시각화
- 지도 LayerControl로 각 polygon 레이어 켜기/끄기
- 왼쪽 패널에 불러온 SHP 데이터 개수 표시

## 다음 고도화

- 기상청 단기예보 조회서비스 API 연동
- 격자 단위 학습 데이터 생성
- RandomForest 기반 침수 위험도 예측
- 최단경로와 안전경로 비교
