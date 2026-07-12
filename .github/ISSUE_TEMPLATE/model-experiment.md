---
name: Model Experiment
about: 모델, Feature, Loss, 하이퍼파라미터 실험
title: ''
labels: experiment
assignees: thisischeese

---

---
name: Model Experiment
about: 모델, Feature, Loss, 하이퍼파라미터 실험
title: "[Experiment] "
labels: "experiment"
assignees: ""
---

## 실험 목적

<!-- 해결하려는 문제와 실험이 필요한 이유를 작성한다. -->

## 가설

<!-- 무엇을 변경하면 어떤 지표가 개선될 것으로 예상하는가? -->

예: GRU hidden size를 128에서 256으로 늘리면 Recall@10이 개선될 것이다.

## Baseline

- 모델:
- 데이터 버전:
- 설정 파일:
- Git commit:
- MLflow Run ID:
- 주요 성능:

## 변경 변수

<!-- 한 번의 실험에서는 가능한 한 하나의 핵심 변수만 변경한다. -->

- 변경 항목:
- 기존 값:
- 실험 값:
- 통제 변수:

## 평가 방법

- 학습 데이터:
- 검증 데이터:
- Random seed:
- 주요 지표:
- 보조 지표:
- 성공 기준:
- 추론 시간 측정 여부:

## 실행 항목

- [ ] 데이터 버전 고정
- [ ] Baseline 재현
- [ ] 실험 실행
- [ ] Metric 및 Artifact 기록
- [ ] 실패 사례 분석
- [ ] 결과 및 결론 작성

## 실험 결과

- MLflow Run ID:
- 주요 성능:
- Baseline 대비 변화:
- Latency:
- GPU 및 실행 환경:

## 결론

<!-- 채택, 기각 또는 추가 실험 필요 여부를 작성한다. -->

- [ ] 채택
- [ ] 기각
- [ ] 추가 실험 필요

## 후속 작업

<!-- 모델 반영, 재실험, 배포 검증 등의 Issue를 연결한다. -->
