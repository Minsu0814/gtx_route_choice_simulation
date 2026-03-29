# Deep Learning 학습 실행 명령어

## CMD에서 실행

```cmd
C:\Users\USER\anaconda3\condabin\conda.bat activate agpu
cd C:\Research\6.route_choice_simulation
python -u code\similarity\deep_learning\train_deep_learning.py --epochs 100 --patience 15
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--models` | 전체 5개 | 특정 모델만 학습 (dnn, tastenet, reslogit, asudnn, lmnl) |
| `--epochs` | 100 | 최대 에폭 수 |
| `--patience` | 15 | Early stopping patience |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 2048 | 배치 크기 |
| `--seed` | 42 | 랜덤 시드 |

## 예시

```cmd
:: 특정 모델만
python -u code\similarity\deep_learning\train_deep_learning.py --models dnn tastenet

:: 에폭/LR 변경
python -u code\similarity\deep_learning\train_deep_learning.py --epochs 200 --lr 5e-4
```

## 환경 정보

- 가상환경: `agpu` (anaconda3)
- GPU: RTX 4070 Ti SUPER 16GB
- PyTorch + CUDA
