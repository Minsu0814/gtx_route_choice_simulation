"""SHAP analysis for Deep Learning Route Choice models."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from module.data import create_dataloaders, MODEL_FEATURES, FEATURE_LABELS, MAX_ALTS
from module.models import (
    DNNChoiceModel, TasteNetModel, ResLogitModel,
    ASUDNNModel, LMNLModel, load_mnl_beta,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'training_set'
DEVICE = 'cpu'  # SHAP은 CPU에서 실행
N_BACKGROUND = 500   # background samples
N_EXPLAIN = 1000     # explain samples

ALL_MODELS = ['dnn', 'tastenet', 'reslogit', 'asudnn', 'lmnl']


def build_model(name, n_features, n_context, mnl_beta=None):
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


def create_model_wrapper(model, test_ds, alt_idx=0):
    """SHAP용 wrapper: numpy (N, n_features) → chosen alt probability."""
    z_fixed = test_ds.z[:N_EXPLAIN].to(DEVICE)
    mask_fixed = test_ds.mask[:N_EXPLAIN].to(DEVICE)
    X_base = test_ds.X[:N_EXPLAIN].clone().to(DEVICE)

    def predict_fn(X_np):
        """X_np: (N, n_features) → probabilities for alt_idx."""
        X_t = X_base.clone()
        X_t[:len(X_np), alt_idx, :] = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            probs = model(X_t[:len(X_np)], z_fixed[:len(X_np)], mask_fixed[:len(X_np)])
        return probs[:, alt_idx].numpy()

    return predict_fn


def create_flat_wrapper(model, test_ds):
    """SHAP용 wrapper: 대안별이 아닌 전체 choice set에서 chosen alt의 확률."""
    def predict_fn(X_np):
        """X_np: (N, n_features) — 각 샘플의 '선택된 대안'의 피처.
        Returns: (N,) — 해당 대안의 선택 확률.
        """
        n = len(X_np)
        X_full = test_ds.X[:n].clone().to(DEVICE)
        z = test_ds.z[:n].to(DEVICE)
        mask = test_ds.mask[:n].to(DEVICE)
        y = test_ds.y[:n]

        # 선택된 대안 인덱스
        chosen_idx = y.argmax(dim=-1)

        # 선택된 대안의 피처를 X_np로 교체
        for i in range(n):
            X_full[i, chosen_idx[i], :] = torch.tensor(X_np[i], dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_full, z, mask)

        # 각 샘플에서 선택된 대안의 확률 반환
        result = probs[torch.arange(n), chosen_idx].numpy()
        return result

    return predict_fn


def run_shap_for_model(model_name, model, test_ds, features, labels):
    """단일 모델에 대해 SHAP 분석 실행."""
    print(f'\n{"="*60}')
    print(f'SHAP Analysis: {model_name.upper()}')
    print(f'{"="*60}')

    model.eval()
    model.to(DEVICE)

    # 선택된 대안의 피처만 추출
    y = test_ds.y[:N_EXPLAIN]
    chosen_idx = y.argmax(dim=-1)
    X_chosen = torch.stack([
        test_ds.X[i, chosen_idx[i], :] for i in range(N_EXPLAIN)
    ]).numpy()

    # Background data
    bg_idx = np.random.choice(N_EXPLAIN, N_BACKGROUND, replace=False)
    background = X_chosen[bg_idx]

    # Wrapper
    predict_fn = create_flat_wrapper(model, test_ds)

    # KernelExplainer
    print(f'  Computing SHAP values (bg={N_BACKGROUND}, explain={N_EXPLAIN})...')
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_chosen, nsamples=100)

    # Mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print(f'\n  Mean |SHAP| (feature importance):')
    print(f'  {"Feature":<25} {"Mean |SHAP|":>12} {"Rank":>6}')
    print(f'  {"-"*45}')

    ranked = np.argsort(-mean_abs_shap)
    for rank, idx in enumerate(ranked):
        print(f'  {labels[idx]:<25} {mean_abs_shap[idx]:>12.6f} {rank+1:>6}')

    # Summary plot
    fig_path = DATA_DIR / f'shap_{model_name}.png'
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_chosen,
        feature_names=labels,
        show=False, max_display=len(features),
    )
    plt.title(f'SHAP Summary — {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved: {fig_path}')

    return mean_abs_shap, shap_values


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--n-background', type=int, default=500)
    parser.add_argument('--n-explain', type=int, default=1000)
    args = parser.parse_args()

    global N_BACKGROUND, N_EXPLAIN
    N_BACKGROUND = args.n_background
    N_EXPLAIN = args.n_explain

    features = list(MODEL_FEATURES)
    labels = list(FEATURE_LABELS)

    # Load data
    print('[1/3] Loading data...')
    _, _, _, test_ds = create_dataloaders(
        data_dir=str(DATA_DIR), batch_size=2048, device=DEVICE, features=features,
    )

    # MNL beta for ResLogit
    import json
    coeff_path = DATA_DIR / 'mnl_coefficients.json'
    mnl_beta = None
    if coeff_path.exists():
        mnl_beta = load_mnl_beta(str(coeff_path), test_ds.scaler, features=features)

    # Run SHAP
    print(f'\n[2/3] Running SHAP for {len(args.models)} model(s)...')
    all_importance = {}

    for model_name in args.models:
        n_features = len(features)
        n_context = test_ds.n_context
        model = build_model(model_name, n_features, n_context, mnl_beta)

        # Load weights
        pt_path = DATA_DIR / f'{model_name}_model.pt'
        if not pt_path.exists():
            print(f'  Skipping {model_name}: {pt_path} not found')
            continue
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))

        mean_abs_shap, _ = run_shap_for_model(
            model_name, model, test_ds, features, labels,
        )
        all_importance[model_name] = mean_abs_shap

    # Comparison table
    print(f'\n[3/3] Feature Importance Comparison (Mean |SHAP|)')
    print(f'{"="*80}')
    header = f'{"Feature":<25}' + ''.join(f'{m:>12}' for m in all_importance.keys())
    print(header)
    print('-' * 80)
    for i, label in enumerate(labels):
        row = f'{label:<25}'
        for m, imp in all_importance.items():
            row += f'{imp[i]:>12.6f}'
        print(row)

    # Comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(all_importance)
    for j, (m, imp) in enumerate(all_importance.items()):
        ax.bar(x + j * width, imp, width, label=m.upper(), alpha=0.8)
    ax.set_xticks(x + width * len(all_importance) / 2)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title('Feature Importance Comparison (SHAP)')
    ax.legend()
    plt.tight_layout()
    fig_path = DATA_DIR / 'shap_comparison.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nComparison plot saved: {fig_path}')
    print('Done.')


if __name__ == '__main__':
    main()
