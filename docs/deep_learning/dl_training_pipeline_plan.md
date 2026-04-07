# Deep Learning Route Choice — 실행 환경 + CLI 구축 계획

## Context

MNL 경로 선택 모델이 완성된 상태에서, 딥러닝 모델 학습/평가 파이프라인을 완성하려 함.
기존 DL 코드(`code/module/deep_learning/`)에 DNN, TasteNet, ResLogit 3개 모델이 이미 구현되어 있고, 노트북도 있음.
**목표**: (1) 한 번의 명령으로 환경 설치, (2) CLI 스크립트로 학습 실행, (3) 최신 모델 추가 검토.

---

## Part 1: 모델 추가 권장 사항

### 기존 3개 모델 평가

| 모델     | 논문                | 특징                          | 판단                |
| -------- | ------------------- | ----------------------------- | ------------------- |
| DNN      | 기본                | Black-box 예측 성능           | ✅ 유지 (baseline)  |
| TasteNet | Han et al., 2022    | β=g(context), OD거리별 이질성 | ✅ 유지 (해석 가능) |
| ResLogit | Wong & Farooq, 2021 | MNL + 잔차 DNN                | ✅ 유지 (MNL 확장)  |

### 추가 권장 모델 (최신 연구 동향 기반)

**1. ASU-DNN (Alternative-Specific Utility DNN)** — 추천

- 출처: Han et al. (2020), "A neural-embedded discrete choice model"
- 핵심: 대안별 별도 sub-network로 utility 학습 → 대안 간 상호작용 포착
- 기존 DNN과의 차이: DNN은 모든 대안에 같은 네트워크 적용, ASU-DNN은 대안별 독립 학습 가능
- 구현 난이도: 낮음 (기존 코드 패턴 재활용)

**2. L-MNL (Learning MNL)** — 추천

- 출처: Sifringer et al. (2020), "Enhancing discrete choice models with representation learning"
- 핵심: DNN이 raw features → latent variables 생성, 이를 MNL의 추가 설명변수로 사용
- TasteNet과의 차이: TasteNet은 β를 context로 변화시키고, L-MNL은 새로운 변수를 학습
- 구현 난이도: 낮음

**3. Transformer/Cross-Attention** — 선택적

- 핵심: 대안 간 self-attention으로 choice set 내 상대적 비교 학습
- 장점: choice set size에 유연, 대안 간 경쟁관계 포착
- 단점: 데이터 규모 대비 과적합 위험, 해석 어려움
- 구현 난이도: 중간

→ **권장**: 기존 3개 + ASU-DNN + L-MNL 추가 (총 5개 모델)

---

## Part 2: 환경 설치 (setup_env.py + requirements.txt)

### 파일 1: `requirements.txt` (프로젝트 루트)

```
numpy>=1.24
pandas>=2.0
pyarrow>=12.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
ijson>=3.2
tqdm>=4.65
```

- PyTorch는 CUDA 버전별 설치 필요 → requirements.txt에서 제외

### 파일 2: `setup_env.py` (프로젝트 루트)

기능:

1. `nvidia-smi` 실행하여 CUDA 버전 자동 감지
2. 감지된 CUDA에 맞는 PyTorch wheel 설치 (cu121/cu118/cpu)
3. `pip install -r requirements.txt`로 나머지 패키지 설치
4. 설치 검증 (torch, pandas, sklearn import 테스트)

사용법:

```bash
python setup_env.py          # 자동 CUDA 감지
python setup_env.py --cpu    # CPU 전용 강제
```

---

## Part 3: CLI 학습 스크립트 (train_deep_learning.py)

### 파일: `code/train_deep_learning.py`

사용법:

```bash
python code/train_deep_learning.py                          # 전체 모델 학습
python code/train_deep_learning.py --models dnn tastenet    # 특정 모델만
python code/train_deep_learning.py --epochs 200 --lr 1e-3   # 하이퍼파라미터
python code/train_deep_learning.py --seed 42 --device cuda  # 재현성 + GPU
```

### CLI 인자:

| 인자           | 기본값 | 설명                                          |
| -------------- | ------ | --------------------------------------------- |
| `--models`     | all    | dnn, tastenet, reslogit, asudnn, lmnl 중 선택 |
| `--epochs`     | 100    | 최대 에폭                                     |
| `--lr`         | 1e-3   | 학습률                                        |
| `--batch-size` | 2048   | 배치 크기                                     |
| `--patience`   | 15     | Early stopping patience                       |
| `--device`     | auto   | cpu/cuda/auto                                 |
| `--seed`       | 42     | 랜덤 시드 (재현성)                            |
| `--data-dir`   | auto   | 학습 데이터 경로                              |

### 실행 흐름:

1. 시드 설정 (torch, numpy, cudnn)
2. `create_dataloaders()` 호출
3. MNL baseline 결과 로드 (`mnl_coefficients.json`)
4. 선택된 모델별 순차 학습 + 평가
5. 비교 테이블 출력 (MNL vs DL 모델들)
6. 결과 JSON + 모델 가중치(.pt) 저장

---

## Part 4: 모델 코드 추가/수정

### 수정 파일: `code/module/deep_learning/dl_models.py`

추가한 클래스:

- `ASUDNNModel`: 대안별 독립 sub-network → utility → masked softmax
- `LMNLModel`: DNN feature extractor → latent vars + linear utility → softmax

### 수정 파일: `code/module/deep_learning/dl_train.py`

- `set_seed(seed)` 함수 추가 (torch, numpy, cudnn deterministic)

### 수정 파일: `code/module/deep_learning/dl_data.py`

- `create_dataloaders()`에 `pin_memory` 자동 설정 (CUDA일 때 True)

---

## Part 5: 구현 순서

| 순서 | 파일                     | 작업                            |
| ---- | ------------------------ | ------------------------------- |
| 1    | `requirements.txt`       | 의존성 목록 생성                |
| 2    | `setup_env.py`           | CUDA 감지 + 환경 설치 스크립트  |
| 3    | `dl_models.py`           | ASU-DNN, L-MNL 모델 클래스 추가 |
| 4    | `dl_train.py`            | `set_seed()` 함수 추가          |
| 5    | `dl_data.py`             | `pin_memory` 자동 설정          |
| 6    | `train_deep_learning.py` | CLI 학습 스크립트 생성          |

---

## 검증 방법

1. **환경 설치**: `python setup_env.py` 실행 → PyTorch + CUDA 정상 인식 확인
2. **CLI 학습**: `python code/train_deep_learning.py --models dnn --epochs 5` (빠른 테스트)
3. **전체 모델**: `python code/train_deep_learning.py` → 5개 모델 학습 + 비교 테이블 출력
4. **결과 파일**: `data/training_set/dl_model_results.json`, `*_model.pt` 생성 확인

---

## 핵심 파일 경로

- `code/module/deep_learning/dl_models.py` — 모델 정의 (수정: ASU-DNN, L-MNL 추가)
- `code/module/deep_learning/dl_data.py` — 데이터 로딩 (수정: pin_memory)
- `code/module/deep_learning/dl_train.py` — 학습 루프 (수정: set_seed)
- `code/train_deep_learning.py` — **새로 생성**: CLI 스크립트
- `requirements.txt` — **새로 생성**: 의존성
- `setup_env.py` — **새로 생성**: 환경 설치
