# RainGuard

공공데이터 기반 도시 침수 위험 예측 및 안전 이동 경로 추천 서비스 MVP입니다.

현재 버전은 제안서 제출 전 구현 가능성을 보여주기 위한 Streamlit 프로토타입입니다. 침수 위험 지점은 `data/sample_risk_points.csv`의 시연용 샘플 데이터이며, 이후 실제 침수흔적도 데이터로 교체할 예정입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 현재 구현

- 강수량 시나리오 선택: `10mm/h`, `30mm/h`, `50mm/h`
- 시나리오별 침수 위험도 카드 표시
- Folium 기반 지도 시각화
- 강수량에 따른 위험 원 색상 변경
- 샘플 위험 지점 CSV 분리

## 다음 고도화

- 서울시 침수흔적도 데이터 반영
- 격자 단위 학습 데이터 생성
- RandomForest 기반 침수 위험도 예측
- 최단경로와 안전경로 비교
