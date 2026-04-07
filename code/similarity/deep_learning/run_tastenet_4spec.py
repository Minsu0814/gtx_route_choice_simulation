"""
TasteNet 4-Spec comparison (ASC dummies, no wait_time_min).
Usage: python run_tastenet_4spec.py [epochs] [batch_size]
"""
import sys, time, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec_config import TN_SPECS, CONTEXT_FEATURES, CATEGORY_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (PROJECT_ROOT / 'data' / 'training_set_new'
             / 'route_choice_filtered_training.parquet')
OUT_PATH = (PROJECT_ROOT / 'data' / 'training_set_new'
            / 'tastenet_4spec.json')
MAX_ALTS = 10

# === Data ===

def load_data(path):
    df = pd.read_parquet(path)
    for col in ['ivt_bus', 'ivt_train', 'ivt_gtx', 'waiting_time',
                'transfer_walk_time', 'in_vehicle_time',
                'access_time', 'egress_time']:
        if col in df.columns:
            df[col] = df[col] / 60.0
    df = df.rename(columns={'waiting_time': 'wait_time_min',
                            'transfer_walk_time': 'transfer_walk_time_min'})
    df['fare_1000won'] = df['fare'] / 1000.0
    df['total_ivt_min'] = df['in_vehicle_time']
    df['ln_access'] = np.log1p(df['access_time'])
    df['ln_egress'] = np.log1p(df['egress_time'])
    df['transport_category'] = df['transport_category'].map(CATEGORY_MAP)
    # ASC dummy columns (reference = bus_only)
    df['is_train_only'] = (df['transport_category'] == 'train_only').astype(float)
    df['is_bus_train'] = (df['transport_category'] == 'bus+train').astype(float)
    df['is_train_gtx'] = (df['transport_category'] == 'train+gtx').astype(float)
    df['choice_prob'] = df['chosen'].astype(float)
    df['n_total'] = 1
    df['choice_set_size'] = df.groupby('od_pair')['od_pair'].transform('count')
    if 'total_distance_km' in df.columns:
        df['od_distance_km'] = (df.groupby('od_pair')
                                ['total_distance_km'].transform('median'))
    elif 'od_distance' in df.columns:
        df['od_distance_km'] = df['od_distance'] / 1000.0
    else:
        df['od_distance_km'] = 10.0
    return df

def split_data(df, test_size=0.2):
    od_dom = df.loc[df.groupby('od_pair')['choice_prob'].idxmax(),
                    ['od_pair', 'transport_category']]
    od_dom = od_dom.set_index('od_pair')['transport_category']
    od_list = np.array(od_dom.index.tolist())
    strat = np.array(od_dom.values.tolist())
    tr, te = train_test_split(od_list, test_size=test_size,
                              random_state=42, stratify=strat)
    return (df[df['od_pair'].isin(set(tr))].copy(),
            df[df['od_pair'].isin(set(te))].copy())

class FilteredDataset(Dataset):
    def __init__(self, df, features, scaler=None, ctx_scaler=None,
                 fit_scaler=False):
        self.n_features = len(features)
        prob_sum = df.groupby('od_pair')['choice_prob'].transform('sum')
        df_v = df[np.abs(prob_sum - 1.0) <= 0.01].copy().sort_values('od_pair')
        od_codes, od_uniq = pd.factorize(df_v['od_pair'], sort=False)
        df_v['_oc'] = od_codes
        df_v['_ai'] = df_v.groupby('_oc').cumcount()
        self.n_groups = len(od_uniq)
        self.max_alts = min(int(df_v['_ai'].max()) + 1, MAX_ALTS)

        X_all = df_v[features].values.astype(np.float64)
        self.scaler = StandardScaler().fit(X_all) if fit_scaler else scaler
        X_sc = self.scaler.transform(X_all).astype(np.float32)

        first = df_v.drop_duplicates('_oc', keep='first')
        z_all = first[CONTEXT_FEATURES].values.astype(np.float64)
        self.ctx_scaler = StandardScaler().fit(z_all) if fit_scaler else ctx_scaler
        z_sc = self.ctx_scaler.transform(z_all).astype(np.float32)

        G, A, F = self.n_groups, self.max_alts, self.n_features
        self.X = torch.zeros(G, A, F)
        self.z = torch.from_numpy(z_sc)
        self.y = torch.zeros(G, A)
        self.mask = torch.zeros(G, A)
        self.weight = torch.from_numpy(first['n_total'].values.astype(np.float32))

        oc, ai = df_v['_oc'].values, df_v['_ai'].values
        valid = ai < self.max_alts
        oc, ai, X_sc = oc[valid], ai[valid], X_sc[valid]
        y_vals = df_v['choice_prob'].values.astype(np.float32)[valid]
        self.X[oc, ai] = torch.from_numpy(X_sc.copy())
        self.y[oc, ai] = torch.from_numpy(y_vals.copy())
        self.mask[oc, ai] = 1.0

    def __len__(self):
        return self.n_groups

    def __getitem__(self, idx):
        return self.X[idx], self.z[idx], self.y[idx], self.mask[idx], self.weight[idx]

# === Model ===

def masked_softmax(logits, mask):
    logits = logits.masked_fill(mask == 0, -1e9)
    return torch.nn.functional.softmax(logits, dim=-1) * mask

class TasteNet(torch.nn.Module):
    def __init__(self, n_features, n_context, hidden_dim=32):
        super().__init__()
        self.taste_net = torch.nn.Sequential(
            torch.nn.Linear(n_context, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_features))

    def forward(self, X, z, mask):
        beta = self.taste_net(z)
        V = (X * beta.unsqueeze(1)).sum(-1)
        return masked_softmax(V, mask)

    def get_betas(self, z):
        with torch.no_grad():
            return self.taste_net(z)

# === Training ===

def train_tastenet(model, train_loader, test_loader, epochs=100,
                   lr=1e-3, patience=15, device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=5, min_lr=1e-6)
    best_loss, best_state, wait, t0 = float('inf'), None, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for X, z, y, mask, w in train_loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            opt.zero_grad()
            probs = model(X, z, mask)
            loss = -(w * (y * torch.log(probs + 1e-10) * mask).sum(-1)).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, z, y, mask, w in test_loader:
                X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
                probs = model(X, z, mask)
                test_loss += -(w * (y * torch.log(probs + 1e-10) * mask).sum(-1)).sum().item()

        sched.step(test_loss)
        if test_loss < best_loss:
            best_loss, best_state, wait = test_loss, deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if epoch <= 5 or epoch % 10 == 0 or wait == 0:
            print(f'  Epoch {epoch:3d}  test_loss={test_loss:,.0f}  '
                  f'lr={opt.param_groups[0]["lr"]:.1e}  wait={wait}  '
                  f'({(time.time()-t0)/60:.1f}m)')
        if wait >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    return model, best_loss

# === Evaluation ===

def evaluate(model, dataset, device='cpu'):
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    model.eval()
    preds, ys, masks, ws = [], [], [], []
    with torch.no_grad():
        for X, z, y, mask, w in loader:
            X, z, y, mask, w = (t.to(device) for t in (X, z, y, mask, w))
            preds.append(model(X, z, mask).cpu())
            ys.append(y.cpu()); masks.append(mask.cpu()); ws.append(w.cpu())

    pred, y, mask, w = [torch.cat(x) for x in [preds, ys, masks, ws]]
    N = pred.shape[0]
    log_p = torch.log(pred + 1e-10)
    ll_beta = (w * (y * log_p * mask).sum(-1)).sum().item()
    ll_0 = -(w.double() * torch.log(mask.sum(-1).double())).sum().item()

    pred_np, y_np = pred.numpy(), y.numpy()
    top1 = sum(pred_np[i, :int(mask[i].sum())].argmax() ==
               y_np[i, :int(mask[i].sum())].argmax() for i in range(N)) / N
    top3 = 0
    for i in range(N):
        nv = int(mask[i].sum())
        if y_np[i, :nv].argmax() in pred_np[i, :nv].argsort()[-min(3, nv):]:
            top3 += 1
    return {'rho_sq': 1 - ll_beta / ll_0, 'top1': top1, 'top3': top3 / N,
            'll_beta': ll_beta, 'll_0': ll_0, 'n_groups': N}

def extract_mean_betas(model, dataset, features, device='cpu'):
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    betas = []
    with torch.no_grad():
        for X, z, y, mask, w in loader:
            betas.append(model.get_betas(z.to(device)).cpu().numpy())
    betas = np.vstack(betas)
    return betas.mean(axis=0), betas.std(axis=0)

# === Main ===

if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'TasteNet 4-Spec (ASC dummies, no wait_time)')
    print(f'Epochs={epochs}, Batch={batch_size}, Device={device}\n')

    df = load_data(DATA_PATH)
    print(f'Loaded: {len(df):,} rows, {df["od_pair"].nunique():,} ODs')
    train_df, test_df = split_data(df)
    print(f'Train: {train_df["od_pair"].nunique():,} ODs, '
          f'Test: {test_df["od_pair"].nunique():,} ODs\n')

    results = []
    for spec_name, features in TN_SPECS.items():
        print(f'\n{"="*70}\n  {spec_name}: {features}\n{"="*70}')
        train_ds = FilteredDataset(train_df, features, fit_scaler=True)
        test_ds = FilteredDataset(test_df, features,
                                  scaler=train_ds.scaler,
                                  ctx_scaler=train_ds.ctx_scaler)
        print(f'  Train: {len(train_ds):,}, Test: {len(test_ds):,}, '
              f'MaxAlts: {train_ds.max_alts}, Features: {len(features)}')

        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model = TasteNet(len(features), len(CONTEXT_FEATURES))
        model, _ = train_tastenet(model, train_ld, test_ld,
                                  epochs=epochs, device=device)

        mt = evaluate(model, test_ds, device)
        mr = evaluate(model, train_ds, device)
        mb, sb = extract_mean_betas(model, test_ds, features, device)

        print(f'\n  Train: rho-sq={mr["rho_sq"]:.4f}, Top-1={mr["top1"]*100:.1f}%')
        print(f'  Test:  rho-sq={mt["rho_sq"]:.4f}, Top-1={mt["top1"]*100:.1f}%, '
              f'Top-3={mt["top3"]*100:.1f}%')
        print(f'\n  {"Feature":<26} {"Mean b":>10} {"Std b":>10}')
        print(f'  {"-"*48}')
        for i, f in enumerate(features):
            print(f'  {f:<26} {mb[i]:>10.4f} {sb[i]:>10.4f} '
                  f'{"(-)" if mb[i] < 0 else "(+)"}')

        results.append({
            'spec': spec_name, 'features': features,
            'train_rho_sq': mr['rho_sq'], 'train_top1': mr['top1'],
            'test_rho_sq': mt['rho_sq'], 'test_top1': mt['top1'],
            'test_top3': mt['top3'], 'test_ll_beta': mt['ll_beta'],
            'test_ll_0': mt['ll_0'],
            'mean_betas': dict(zip(features, mb.tolist())),
            'std_betas': dict(zip(features, sb.tolist())),
        })

    # Summary
    print(f'\n\n{"="*80}\nTASTENET 4-SPEC SUMMARY\n{"="*80}')
    print(f'{"Spec":<25} {"Train r2":>10} {"Test r2":>10} '
          f'{"Test Top-1":>10} {"Test Top-3":>10}')
    print(f'{"-"*70}')
    for r in results:
        print(f'{r["spec"]:<25} {r["train_rho_sq"]:>10.4f} '
              f'{r["test_rho_sq"]:>10.4f} {r["test_top1"]*100:>9.1f}% '
              f'{r["test_top3"]*100:>9.1f}%')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {OUT_PATH}')
