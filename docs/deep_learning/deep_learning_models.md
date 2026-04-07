# Deep Learning Route Choice Models

MNL 대비 딥러닝 경로 선택 모델 성능 비교.

## 모델 구조

### 1. DNN (Deep Neural Network)

- **구조**: Linear(11->64) -> ReLU -> Dropout -> Linear(64->32) -> ReLU -> Dropout -> Linear(32->1)
- **원리**: 각 대안의 피처를 비선형 변환하여 utility score 산출 -> masked softmax
- **특징**: 순수 예측 성능 극대화, 해석 불가
- **파라미터**: 2,881개

### 2. TasteNet

- **구조**: taste_net: Linear(n_context->32) -> ReLU -> Linear(32->11)
- **원리**: context(OD 거리, choice set 크기) -> beta(z) -> V = X \* beta(z)
- **특징**: OD 거리에 따라 시간/비용 감수성이 달라지는 것을 학습 (해석 가능)
- **파라미터**: 459개

### 3. ResLogit

- **구조**: V = X \* beta_MNL + DNN_residual(X)
- **원리**: MNL 선형 유틸리티 위에 비선형 잔차 학습
- **특징**: MNL beta로 초기화, 잔차 비중으로 MNL 충분성 평가 가능 (해석 가능)
- **파라미터**: 428개 (trainable)

## 공통 설계

| 항목        | 설정                                                                                           |
| ----------- | ---------------------------------------------------------------------------------------------- |
| 피처        | MNL과 동일 11개 (IVT, Wait, Access/Egress/Transfer walk, Dist, Transfers, Fare, Bus/Train/GTX) |
| 정규화      | StandardScaler (train set 기준)                                                                |
| 패딩        | MAX_ALTS=5, mask로 유효 대안 구분                                                              |
| 손실함수    | Weighted cross-entropy (n_total 가중치)                                                        |
| 최적화      | Adam + ReduceLROnPlateau + Early stopping                                                      |
| 데이터 분할 | OD-level stratified (80/20, random_state=42)                                                   |

## 파일 구조

```
code/
  module/
    dl_data.py          # RouteChoiceDataset + DataLoader
    dl_models.py        # DNN, TasteNet, ResLogit 모델 정의
    dl_train.py         # 학습 루프 + 평가 함수
  route_choice_deep_learning.ipynb  # 메인 실험 노트북
```

## 평가 메트릭 (MNL과 동일)

- McFadden rho-squared: 1 - LL(beta) / LL(0)
- FPR Top-1: 최고 확률 대안이 실제 선택과 일치하는 비율
- FPR Top-3: 상위 3개 대안에 실제 선택이 포함되는 비율
- RMSE: 예측 확률과 실제 choice_prob의 RMSE

## 검증 기준

- LL(0) 값이 MNL과 동일 (train: -37,222,390.89)
- Train/Test OD 수 일치 (478,345 / 119,587)
- 각 모델의 loss convergence 확인

## 참고 문헌

- TasteNet: Sifringer et al. (2020) "Enhancing discrete choice models with representation learning"
- ResLogit: Wong & Farooq (2021) "ResLogit: A residual neural network logit model for data-driven choice modelling"
