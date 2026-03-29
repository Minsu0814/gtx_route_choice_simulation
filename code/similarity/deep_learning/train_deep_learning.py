"""
Deep Learning Route Choice — CLI 학습 스크립트.

Usage:
    python code/train_deep_learning.py                          # 전체 모델 학습
    python code/train_deep_learning.py --models dnn tastenet    # 특정 모델만
    python code/train_deep_learning.py --epochs 200 --lr 1e-3   # 하이퍼파라미터
    python code/train_deep_learning.py --seed 42 --device cuda  # 재현성 + GPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from module.data import (
    create_dataloaders, MODEL_FEATURES, FEATURE_LABELS, MAX_ALTS,
)
from module.models import (
    DNNChoiceModel, TasteNetModel, ResLogitModel,
    ASUDNNModel, LMNLModel, load_mnl_beta,
)
from module.train import (
    set_seed, train_model, evaluate_model, print_metrics,
)

ALL_MODELS = ['dnn', 'tastenet', 'reslogit', 'asudnn', 'lmnl']

DATA_DIR_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'training_set',
    Path(__file__).resolve().parent / '..' / '..' / '..' / 'data' / 'training_set',
]


def resolve_data_dir(user_dir=None):
    """데이터 디렉토리 탐색."""
    if user_dir:
        p = Path(user_dir)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f'Data directory not found: {user_dir}')

    for p in DATA_DIR_CANDIDATES:
        p = p.resolve()
        if (p / 'route_choice_training.parquet').exists():
            return str(p)

    raise FileNotFoundError(
        'Cannot find route_choice_training.parquet. '
        'Use --data-dir to specify the path.'
    )


def resolve_device(user_device):
    """디바이스 결정."""
    if user_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return user_device


def build_model(name, n_features, n_context, mnl_beta=None):
    """모델 이름으로 인스턴스 생성."""
    if name == 'dnn':
        return DNNChoiceModel(n_features)
    elif name == 'tastenet':
        return TasteNetModel(n_features, n_context)
    elif name == 'reslogit':
        return ResLogitModel(n_features, mnl_beta=mnl_beta)
    elif name == 'asudnn':
        return ASUDNNModel(n_features, max_alts=MAX_ALTS)
    elif name == 'lmnl':
        return LMNLModel(n_features)
    else:
        raise ValueError(f'Unknown model: {name}')


def load_mnl_results(data_dir):
    """MNL baseline 결과 로드."""
    coeff_path = Path(data_dir) / 'mnl_coefficients.json'
    if not coeff_path.exists():
        return None
    with open(coeff_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_comparison_table(results, mnl_results):
    """MNL vs DL 모델 비교 테이블 출력."""
    print(f'\n{"=" * 80}')
    print('Model Comparison (Test Set)')
    print(f'{"=" * 80}')

    header = f'{"Model":<12} {"ρ²":>8} {"FPR Top-1":>10} {"FPR Top-3":>10} {"RMSE":>8} {"LL(β)":>14}'
    print(header)
    print('-' * 80)

    # MNL baseline
    if mnl_results:
        ts = mnl_results.get('train_stats', {})
        print(f'{"MNL":<12} {ts.get("rho_squared", 0):>8.4f} {"-":>10} {"-":>10} {"-":>8} {ts.get("ll_beta", 0):>14.0f}')

    # DL models
    for name, metrics in results.items():
        print(
            f'{name:<12} '
            f'{metrics["rho_sq"]:>8.4f} '
            f'{metrics["fpr_top1"]:>10.4f} '
            f'{metrics["fpr_top3"]:>10.4f} '
            f'{metrics["rmse"]:>8.4f} '
            f'{metrics["ll_beta"]:>14.0f}'
        )

    print(f'{"=" * 80}')


def extract_and_save_betas(model, model_name, features, scaler, context_scaler,
                           metrics, data_dir, test_ds=None):
    """모델에서 β 추출 → JSON 저장 (MNL과 동일 포맷).

    - TasteNet/ResLogit/L-MNL: 직접 β 추출
    - DNN/ASU-DNN: gradient 기반 pseudo-β (test_ds 필요)
    """
    sigma = scaler.scale_.astype(np.float64)

    betas_scaled = None
    method = 'direct'
    model_cpu = model.cpu()
    model_cpu.eval()

    if model_name == 'tastenet':
        z_mean = torch.zeros(1, len(context_scaler.mean_))
        betas_scaled = model_cpu.get_betas(z_mean).squeeze(0).numpy().astype(np.float64)
    elif model_name == 'reslogit':
        betas_scaled = model_cpu.beta_mnl.detach().numpy().astype(np.float64)
    elif model_name == 'lmnl':
        all_weights = model_cpu.get_betas()
        betas_scaled = all_weights[:len(features)].astype(np.float64)
    elif model_name in ('dnn', 'asudnn') and test_ds is not None:
        method = 'gradient'
        betas_scaled = _compute_gradient_betas(model_cpu, test_ds, len(features))
    else:
        return

    betas_raw = betas_scaled / sigma
    _save_beta_json(model_name, features, betas_scaled, betas_raw, metrics,
                    data_dir, method=method)


def _compute_gradient_betas(model, test_ds, n_features):
    """Test set 평균 gradient ∂V/∂x 계산 → pseudo-β (scaled space)."""
    grad_sum = np.zeros(n_features, dtype=np.float64)
    n_valid = 0
    batch_size = 4096

    for start in range(0, len(test_ds), batch_size):
        end = min(start + batch_size, len(test_ds))
        X_batch = test_ds.X[start:end].clone()
        z_batch = test_ds.z[start:end]
        mask_batch = test_ds.mask[start:end]

        X_batch.requires_grad_(True)
        probs = model(X_batch, z_batch, mask_batch)

        # 최대 확률 대안에 대한 gradient
        chosen = probs.argmax(dim=-1)
        chosen_probs = probs[torch.arange(len(chosen)), chosen]
        chosen_probs.sum().backward()

        grad = X_batch.grad
        valid_mask = mask_batch.unsqueeze(-1).bool()
        grad_masked = grad * valid_mask.float()

        grad_sum += grad_masked.sum(dim=(0, 1)).detach().numpy().astype(np.float64)
        n_valid += valid_mask.sum().item() // n_features

    return grad_sum / n_valid


def _save_beta_json(model_name, features, betas_scaled, betas_raw, metrics,
                    data_dir, method='direct'):
    """β를 JSON으로 저장."""
    beta_dict = {f: float(betas_raw[i]) for i, f in enumerate(features)}

    dtumos = {
        'beta_time': beta_dict.get('total_ivt_min', 0.0),
        'beta_cost': beta_dict.get('fare_1000won', 0.0),
        'beta_walk_access_log': beta_dict.get('ln_access', 0.0),
        'beta_walk_egress_log': beta_dict.get('ln_egress', 0.0),
        'beta_walk_transfer': beta_dict.get('transfer_walk_time_min', 0.0),
        'beta_transfer': beta_dict.get('num_transfers', 0.0),
    }

    out = {
        'model': model_name.upper(),
        'method': method,
        'features': features,
        'beta': beta_dict,
        'beta_scaled': {f: float(betas_scaled[i]) for i, f in enumerate(features)},
        'rho_squared': float(metrics['rho_sq']),
        'dtumos_mapping': dtumos,
    }

    out_path = Path(data_dir) / f'{model_name}_coefficients.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  Betas ({method}) saved to {out_path}')


def save_results(results, data_dir):
    """결과 JSON 저장."""
    out_path = Path(data_dir) / 'dl_model_results.json'

    serializable = {}
    for name, metrics in results.items():
        m = {}
        for k, v in metrics.items():
            if k == 'top3_by_size':
                m[k] = {str(cs): vals for cs, vals in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                m[k] = float(v)
            else:
                m[k] = v
        serializable[name] = m

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Deep Learning Route Choice Model Training'
    )
    parser.add_argument(
        '--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS,
        help=f'Models to train (default: all). Choices: {", ".join(ALL_MODELS)}'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size (default: 2048)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'], help='Device (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--data-dir', default=None, help='Training data directory')
    parser.add_argument('--no-fare', action='store_true', help='Exclude fare from features')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = resolve_data_dir(args.data_dir)

    # Feature selection
    features = list(MODEL_FEATURES)
    if args.no_fare:
        features = [f for f in features if f != 'fare']

    print('=' * 60)
    print('Deep Learning Route Choice - Training')
    print('=' * 60)
    print(f'  Models:     {", ".join(args.models)}')
    print(f'  Device:     {device}')
    print(f'  Epochs:     {args.epochs}')
    print(f'  LR:         {args.lr}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Patience:   {args.patience}')
    print(f'  Seed:       {args.seed}')
    print(f'  Data dir:   {data_dir}')
    print(f'  Fare:       {"excluded" if args.no_fare else "included"}')
    print()

    # Load data
    print('[1/4] Loading data...')
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, device=device,
        features=features,
    )

    n_features = train_ds.n_features
    n_context = train_ds.n_context

    # Load MNL baseline
    print('\n[2/4] Loading MNL baseline...')
    mnl_results = load_mnl_results(data_dir)
    mnl_beta = None
    if mnl_results:
        coeff_path = Path(data_dir) / 'mnl_coefficients.json'
        mnl_beta = load_mnl_beta(str(coeff_path), train_ds.scaler, features=features)
        print(f'  MNL ρ² = {mnl_results["train_stats"]["rho_squared"]:.4f}')
    else:
        print('  MNL coefficients not found - ResLogit will use zero init')

    # Train models
    print(f'\n[3/4] Training {len(args.models)} model(s)...')
    all_results = {}
    model_states = {}
    t_total = time.time()

    for model_name in args.models:
        print(f'\n{"-" * 60}')
        print(f'Training: {model_name.upper()}')
        print(f'{"-" * 60}')

        model = build_model(model_name, n_features, n_context, mnl_beta)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Parameters: {n_params:,}')

        t0 = time.time()
        history, best_state = train_model(
            model, train_loader, test_loader, train_ds, test_ds,
            lr=args.lr, epochs=args.epochs, patience=args.patience,
            device=device, verbose=True,
        )
        elapsed = time.time() - t0
        print(f'  Training time: {elapsed:.1f}s')

        # Evaluate
        metrics, preds, targets, masks, weights = evaluate_model(
            model, test_loader, test_ds, device=device,
        )
        print_metrics(model_name.upper(), metrics)

        all_results[model_name] = metrics
        model_states[model_name] = best_state

        # Save model weights
        pt_path = Path(data_dir) / f'{model_name}_model.pt'
        torch.save(best_state, pt_path)
        print(f'  Model saved to {pt_path}')

        # Extract betas
        extract_and_save_betas(
            model, model_name, features, train_ds.scaler,
            train_ds.context_scaler, metrics, data_dir, test_ds=test_ds,
        )

    total_elapsed = time.time() - t_total

    # Summary
    print(f'\n[4/4] Summary (total: {total_elapsed:.1f}s)')
    print_comparison_table(all_results, mnl_results)

    # Save results
    save_results(all_results, data_dir)


if __name__ == '__main__':
    main()
