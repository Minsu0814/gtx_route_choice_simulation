"""
Deep Learning Route Choice — 학습 루프 + 평가 함수.

- Weighted cross-entropy loss (n_total 가중치)
- Adam + ReduceLROnPlateau + Early stopping
- 평가: McFadden ρ², FPR Top-1/Top-3, RMSE
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import random
from copy import deepcopy


def set_seed(seed=42):
    """재현성을 위한 시드 설정 (torch, numpy, cudnn)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def weighted_cross_entropy(pred_probs, target_probs, mask, weights):
    """가중 Cross-Entropy 손실 (choice_prob 기반).

    L = -Σ_i w_i Σ_j y_ij log(p_ij)   (j = valid alternatives only)

    Args:
        pred_probs:   (B, A) — 모델 예측 확률 (masked softmax 출력)
        target_probs: (B, A) — 실제 choice_prob
        mask:         (B, A) — 유효 대안=1
        weights:      (B,)   — n_total per OD
    Returns:
        scalar loss
    """
    log_probs = torch.log(pred_probs + 1e-10)
    ce_per_sample = -(target_probs * log_probs * mask).sum(dim=-1)  # (B,)
    return (weights * ce_per_sample).sum()


def compute_ll_null(dataset):
    """LL(0): 균등 확률 모델의 log-likelihood (MNL과 동일 방식)."""
    n_alts = dataset.mask.sum(dim=-1).double()  # (N,)
    weights = dataset.weight.double()            # (N,)
    return -(weights * torch.log(n_alts)).sum().item()


def train_model(model, train_loader, test_loader, train_ds, test_ds,
                lr=1e-3, epochs=100, patience=10, device='cpu',
                verbose=True):
    """모델 학습 + early stopping.

    Returns:
        history: dict with train_loss, test_loss per epoch
        best_model_state: 최적 모델 가중치
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, min_lr=1e-6)

    best_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'test_loss': [], 'lr': []}

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X, z, y, mask, w in train_loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            optimizer.zero_grad()
            probs = model(X, z, mask)
            loss = weighted_cross_entropy(probs, y, mask, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss

        # --- Evaluate ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, z, y, mask, w in test_loader:
                X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
                probs = model(X, z, mask)
                loss = weighted_cross_entropy(probs, y, mask, w)
                test_loss += loss.item()

        # Scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(current_lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if verbose and (epoch <= 5 or epoch % 10 == 0 or wait == 0):
            elapsed = time.time() - t0
            print(f'  Epoch {epoch:3d} | '
                  f'Train Loss: {train_loss/1e6:.4f}M | '
                  f'Test Loss: {test_loss/1e6:.4f}M | '
                  f'LR: {current_lr:.1e} | '
                  f'{"*best*" if wait == 0 else f"wait {wait}"} | '
                  f'{elapsed:.0f}s')

        if wait >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch}')
            break

    # Restore best model
    model.load_state_dict(best_state)
    return history, best_state


def evaluate_model(model, test_loader, test_ds, device='cpu'):
    """MNL과 동일한 메트릭으로 평가.

    Returns:
        metrics: dict with ll_beta, ll_0, rho_sq, fpr_top1, fpr_top3, rmse
        all_preds: (N, MAX_ALTS) numpy
        all_targets: (N, MAX_ALTS) numpy
        all_masks: (N, MAX_ALTS) numpy
        all_weights: (N,) numpy
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_masks = []
    all_weights = []

    total_loss = 0.0

    with torch.no_grad():
        for X, z, y, mask, w in test_loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            probs = model(X, z, mask)
            loss = weighted_cross_entropy(probs, y, mask, w)
            total_loss += loss.item()

            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_weights.append(w.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    weights = np.concatenate(all_weights, axis=0)

    n_ods = len(preds)

    # LL(β) = -total_loss (weighted)
    ll_beta = -total_loss

    # LL(0): 균등 확률
    ll_0 = 0.0
    for i in range(n_ods):
        n_alts = int(masks[i].sum())
        ll_0 -= weights[i] * np.log(n_alts)

    # McFadden ρ²
    rho_sq = 1 - ll_beta / ll_0

    # Top-1 Accuracy (FPR)
    top1_correct = 0
    for i in range(n_ods):
        n_alts = int(masks[i].sum())
        if np.argmax(preds[i, :n_alts]) == np.argmax(targets[i, :n_alts]):
            top1_correct += 1
    fpr_top1 = top1_correct / n_ods

    # Top-3 Accuracy
    top3_correct = 0
    top3_by_size = {}
    for i in range(n_ods):
        n_alts = int(masks[i].sum())
        k = min(3, n_alts)
        top_k = np.argsort(preds[i, :n_alts])[-k:]
        actual_best = np.argmax(targets[i, :n_alts])
        hit = actual_best in top_k

        if n_alts not in top3_by_size:
            top3_by_size[n_alts] = [0, 0]
        top3_by_size[n_alts][1] += 1
        if hit:
            top3_correct += 1
            top3_by_size[n_alts][0] += 1

    fpr_top3 = top3_correct / n_ods

    # RMSE (valid alternatives only)
    valid = masks.astype(bool)
    rmse = np.sqrt(np.mean((preds[valid] - targets[valid]) ** 2))

    metrics = {
        'll_beta': ll_beta,
        'll_0': ll_0,
        'rho_sq': rho_sq,
        'fpr_top1': fpr_top1,
        'fpr_top3': fpr_top3,
        'top1_count': f'{top1_correct}/{n_ods}',
        'top3_count': f'{top3_correct}/{n_ods}',
        'rmse': rmse,
        'n_ods': n_ods,
        'top3_by_size': top3_by_size,
    }

    return metrics, preds, targets, masks, weights


def print_metrics(name, metrics):
    """평가 결과 출력."""
    print(f'\n{"=" * 60}')
    print(f'{name} - Test Set Evaluation')
    print(f'{"=" * 60}')
    print(f'  LL(0):          {metrics["ll_0"]:.4f}')
    print(f'  LL(beta):       {metrics["ll_beta"]:.4f}')
    print(f'  McFadden rho2:  {metrics["rho_sq"]:.4f}')
    print(f'  FPR Top-1:      {metrics["fpr_top1"]:.4f} ({metrics["top1_count"]})')
    print(f'  FPR Top-3:      {metrics["fpr_top3"]:.4f} ({metrics["top3_count"]})')
    print(f'  RMSE:           {metrics["rmse"]:.4f}')
    print(f'  N (ODs):        {metrics["n_ods"]:,}')

    if 'top3_by_size' in metrics:
        print(f'\n  Top-3 by choice set size:')
        for cs in sorted(metrics['top3_by_size']):
            hit, total = metrics['top3_by_size'][cs]
            pct = hit / total * 100
            print(f'    size={cs}: {pct:5.1f}% ({hit}/{total})')
