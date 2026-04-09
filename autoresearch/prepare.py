"""
Route Choice AutoResearch - Data + Evaluation (FIXED LAYER).
Do NOT modify this file.
"""
import sys, os, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'similarity' / 'deep_learning'))
from module.data import (load_and_split, RouteChoiceDataset, create_dataloaders,
                         MODEL_FEATURES, CONTEXT_FEATURES, MAX_ALTS)

DATA_DIR = str(PROJECT_ROOT / 'data' / 'training_set')
MNL_COEFF_PATH = PROJECT_ROOT / 'data' / 'training_set' / 'mnl_k3_coefficients.json'
RESULTS_TSV = Path(__file__).resolve().parent / 'results.tsv'
BASELINE_JSON = Path(__file__).resolve().parent / 'baseline.json'
N_FEATURES = len(MODEL_FEATURES)
N_CONTEXT = len(CONTEXT_FEATURES)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(batch_size=2048):
    return create_dataloaders(data_dir=DATA_DIR, batch_size=batch_size, device=get_device())

def load_mnl_beta(scaler):
    if not MNL_COEFF_PATH.exists(): return None
    with open(MNL_COEFF_PATH, 'r', encoding='utf-8') as f: coeff = json.load(f)
    beta_raw = np.array([coeff['beta'][feat] for feat in MODEL_FEATURES], dtype=np.float32)
    return beta_raw * scaler.scale_.astype(np.float32)

def masked_softmax(logits, mask):
    logits = logits.masked_fill(mask == 0, -1e9)
    return F.softmax(logits, dim=-1) * mask

def weighted_cross_entropy(pred_probs, target_probs, mask, weights):
    log_probs = torch.log(pred_probs + 1e-10)
    ce = -(target_probs * log_probs * mask).sum(dim=-1)
    return (weights * ce).sum()

def evaluate(model, test_loader, test_ds, device='cpu'):
    model.eval(); model = model.to(device)
    all_p, all_t, all_m, all_w = [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, z, y, mask, w in test_loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            probs = model(X, z, mask)
            total_loss += weighted_cross_entropy(probs, y, mask, w).item()
            all_p.append(probs.cpu().numpy()); all_t.append(y.cpu().numpy())
            all_m.append(mask.cpu().numpy()); all_w.append(w.cpu().numpy())
    preds = np.concatenate(all_p); targets = np.concatenate(all_t)
    masks = np.concatenate(all_m); weights = np.concatenate(all_w)
    n = len(preds)
    ll_beta = -total_loss
    ll_0 = sum(-weights[i] * np.log(int(masks[i].sum())) for i in range(n))
    rho_sq = 1 - ll_beta / ll_0
    top1 = sum(np.argmax(preds[i,:int(masks[i].sum())]) == np.argmax(targets[i,:int(masks[i].sum())]) for i in range(n))
    top3 = 0
    for i in range(n):
        na = int(masks[i].sum()); k = min(3, na)
        if np.argmax(targets[i,:na]) in np.argsort(preds[i,:na])[-k:]: top3 += 1
    valid = masks.astype(bool)
    rmse = np.sqrt(np.mean((preds[valid] - targets[valid]) ** 2))
    return {'rho_sq': round(float(rho_sq), 6), 'fpr_top1': round(float(top1/n), 4),
            'fpr_top3': round(float(top3/n), 4), 'rmse': round(float(rmse), 4),
            'll_beta': round(float(ll_beta), 2), 'll_0': round(float(ll_0), 2), 'n_ods': int(n)}

def train_model(model, train_loader, test_loader, train_ds, test_ds,
                lr=1e-3, epochs=100, patience=15, device='cpu',
                time_budget=300, verbose=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    best_loss, best_state, wait = float('inf'), None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        elapsed = time.time() - t0
        if elapsed > time_budget:
            if verbose: print(f'  Time budget ({time_budget}s) reached at epoch {epoch}')
            break
        model.train(); total_loss = 0.0
        for X, z, y, mask, w in train_loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            optimizer.zero_grad()
            probs = model(X, z, mask)
            loss = weighted_cross_entropy(probs, y, mask, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step(); total_loss += loss.item()
        model.eval(); test_loss = 0.0
        with torch.no_grad():
            for X, z, y, mask, w in test_loader:
                X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
                test_loss += weighted_cross_entropy(model(X, z, mask), y, mask, w).item()
        scheduler.step(test_loss)
        lr_now = optimizer.param_groups[0]['lr']
        if test_loss < best_loss:
            best_loss = test_loss; best_state = deepcopy(model.state_dict()); wait = 0
        else: wait += 1
        if verbose and (epoch <= 5 or epoch % 10 == 0 or wait == 0):
            print(f'  Epoch {epoch:3d} | Train: {total_loss/1e6:.4f}M | Test: {test_loss/1e6:.4f}M | LR: {lr_now:.1e} | {"*best*" if wait==0 else f"wait {wait}"} | {elapsed:.0f}s')
        if wait >= patience:
            if verbose: print(f'  Early stopping at epoch {epoch}')
            break
    model.load_state_dict(best_state)
    return best_state, best_loss, epoch

def log_result(exp, model_name, metrics, kept, notes=''):
    import datetime
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, 'w') as f:
            f.write('timestamp\texperiment\tmodel\trho_sq\tfpr_top1\tfpr_top3\trmse\tkept\tnotes\n')
    with open(RESULTS_TSV, 'a') as f:
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{ts}\t{exp}\t{model_name}\t{metrics["rho_sq"]:.6f}\t{metrics["fpr_top1"]:.4f}\t{metrics["fpr_top3"]:.4f}\t{metrics["rmse"]:.4f}\t{"Y" if kept else "N"}\t{notes}\n')

def load_baseline():
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, 'r') as f: return json.load(f)
    return None

def save_baseline(metrics, model_name, experiment_name):
    data = {'model_name': model_name, 'experiment': experiment_name}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)): data[k] = float(v)
        elif isinstance(v, (np.integer, int)): data[k] = int(v)
        else: data[k] = v
    with open(BASELINE_JSON, 'w') as f: json.dump(data, f, indent=2)

def print_comparison(baseline, new_metrics):
    if baseline is None:
        print('\n  No baseline yet - this will be the first.')
        return
    print(f'\n  {"Metric":<12} {"Baseline":>10} {"New":>10} {"Delta":>10} {"Result":>8}')
    print(f'  {"-"*52}')
    for key in ['rho_sq', 'fpr_top1', 'fpr_top3', 'rmse']:
        old, new = baseline[key], new_metrics[key]
        delta = new - old
        better = (delta > 0) if key != 'rmse' else (delta < 0)
        print(f'  {key:<12} {old:>10.4f} {new:>10.4f} {delta:>+10.4f} {"BETTER" if better else "worse":>8}')

if __name__ == '__main__':
    print('=== Route Choice AutoResearch: Data Check ===')
    set_seed(42); device = get_device(); print(f'Device: {device}')
    train_loader, test_loader, train_ds, test_ds = load_data()
    print(f'Train: {len(train_ds):,} ODs, Test: {len(test_ds):,} ODs')
    print(f'Features: {N_FEATURES}, Context: {N_CONTEXT}, MAX_ALTS: {MAX_ALTS}')
    b = load_baseline()
    if b: print(f'\nBaseline: {b["model_name"]} (rho_sq={b["rho_sq"]:.4f})')
    else: print('\nNo baseline yet. Run train.py to create one.')
    print('\nData check OK!')
