"""
Deep Learning Route Choice — Dataset & DataLoader.

RouteChoiceDataset: 가변 choice set(2~5) → MAX_ALTS=5 패딩 + 마스킹.
MNL 노트북과 동일한 OD-level stratified split (random_state=42).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAX_ALTS = 5

# MNL K3 사양과 동일한 9개 피처
MODEL_FEATURES = [
    'total_ivt_min',
    'ln_access', 'ln_egress',
    'transfer_walk_time_min',
    'num_transfers',
    'fare_1000won',
    'has_bus', 'has_train', 'has_gtx',
]

FEATURE_LABELS = [
    'Total IVT (min)',
    'ln(1+Access walk)', 'ln(1+Egress walk)',
    'Transfer walk (min)',
    'Transfers',
    'Fare (1000won)',
    'Has bus', 'Has train', 'Has GTX',
]

# Context features for TasteNet
CONTEXT_FEATURES = ['od_distance_km', 'choice_set_size']


def load_and_split(data_dir='../../../data/choice_set/java_default', test_size=0.2, random_state=42):
    """MNL과 동일한 OD-level stratified split으로 데이터 로드."""
    data_path = Path(data_dir) / 'route_choice_training.parquet'
    df = pd.read_parquet(data_path)

    # 파생 피처 생성 (K3 사양)
    for col in ['access_time', 'egress_time', 'transfer_walk_time']:
        df[col + '_min'] = df[col] / 60
    df['in_vehicle_time_min'] = df['in_vehicle_time'] / 60
    df['total_ivt_min'] = df['in_vehicle_time_min']  # 통합 IVT
    df['ln_access'] = np.log1p(df['access_time_min'])
    df['ln_egress'] = np.log1p(df['egress_time_min'])
    df['fare_1000won'] = df['fare'] / 1000

    # Context features
    if 'od_distance' in df.columns:
        df['od_distance_km'] = df['od_distance'] / 1000
    else:
        # OD 거리 = 각 OD의 total_distance 중앙값으로 대체
        df['od_distance_km'] = df.groupby('od_pair')['total_distance_km'].transform('median')

    # OD별 대표 수단 → stratified split
    od_dominant = df.loc[
        df.groupby('od_pair')['choice_prob'].idxmax(),
        ['od_pair', 'transport_category']
    ].set_index('od_pair')['transport_category']

    def coarsen(cat):
        return 'gtx_related' if 'gtx' in cat else cat

    od_strat = od_dominant.map(coarsen)
    od_list = np.array(od_strat.index.tolist())
    strat_labels = np.array(od_strat.values.tolist())

    train_ods, test_ods = train_test_split(
        od_list, test_size=test_size,
        random_state=random_state, stratify=strat_labels
    )

    train_od_set = set(train_ods)
    test_od_set = set(test_ods)

    # GTX OD 보정: 9007↔9008은 반드시 test로 이동 (MNL과 동일)
    gtx_move_to_test = ['9007_9008', '9008_9007']
    for od in gtx_move_to_test:
        train_od_set.discard(od)
        test_od_set.add(od)

    train_df = df[df['od_pair'].isin(train_od_set)].copy()
    test_df = df[df['od_pair'].isin(test_od_set)].copy()

    return train_df, test_df


class RouteChoiceDataset(Dataset):
    """
    각 샘플 = 하나의 OD (choice set).
    패딩하여 MAX_ALTS 크기로 맞추고, mask로 유효 대안 표시.

    Returns:
        X:      (MAX_ALTS, n_features) — 피처 행렬 (정규화됨)
        z:      (n_context,) — context 피처 (TasteNet용)
        y:      (MAX_ALTS,) — choice_prob (정답)
        mask:   (MAX_ALTS,) — 유효 대안=1, 패딩=0
        weight: scalar — n_total (가중치)
    """

    def __init__(self, df, scaler=None, context_scaler=None, fit_scaler=False,
                 features=None):
        features = features or MODEL_FEATURES
        self.n_features = len(features)
        self.n_context = len(CONTEXT_FEATURES)

        # choice_prob 합이 1이 아닌 OD 제거 (벡터화)
        prob_sum = df.groupby('od_pair')['choice_prob'].transform('sum')
        df_valid = df[np.abs(prob_sum - 1.0) <= 0.01].copy()

        # OD별 내부 인덱스 (0, 1, 2, ...) 부여
        df_valid = df_valid.sort_values('od_pair')
        od_codes, od_uniq = pd.factorize(df_valid['od_pair'], sort=False)
        df_valid['_od_code'] = od_codes
        df_valid['_alt_idx'] = df_valid.groupby('_od_code').cumcount()

        self.n_groups = len(od_uniq)

        # Scaler fitting (벡터화)
        X_all = np.array(df_valid[features].values, dtype=np.float64)
        if fit_scaler:
            self.scaler = StandardScaler().fit(X_all)
        else:
            self.scaler = scaler

        X_scaled = self.scaler.transform(X_all).astype(np.float32)

        # Context scaler
        first_rows = df_valid.drop_duplicates('_od_code', keep='first')
        z_all = first_rows[CONTEXT_FEATURES].values.astype(np.float64)
        if fit_scaler:
            self.context_scaler = StandardScaler().fit(z_all)
        else:
            self.context_scaler = context_scaler

        z_scaled = self.context_scaler.transform(z_all).astype(np.float32)

        # Pre-allocate tensors (zero-padded)
        self.X = torch.zeros(self.n_groups, MAX_ALTS, self.n_features)
        self.z = torch.from_numpy(z_scaled)
        self.y = torch.zeros(self.n_groups, MAX_ALTS)
        self.mask = torch.zeros(self.n_groups, MAX_ALTS)
        self.weight = torch.from_numpy(
            np.array(first_rows['n_total'].values, dtype=np.float32)
        )

        # 벡터화 텐서 채우기: scatter로 (od_code, alt_idx) 위치에 값 배치
        oc = df_valid['_od_code'].values
        ai = df_valid['_alt_idx'].values
        y_vals = np.array(df_valid['choice_prob'].values, dtype=np.float32)

        self.X[oc, ai] = torch.from_numpy(X_scaled.copy())
        self.y[oc, ai] = torch.from_numpy(y_vals.copy())
        self.mask[oc, ai] = 1.0

    def __len__(self):
        return self.n_groups

    def __getitem__(self, idx):
        return (self.X[idx], self.z[idx], self.y[idx],
                self.mask[idx], self.weight[idx])


def print_mode_share(train_df, test_df):
    """Train/Test 수단 분담률 출력."""
    for label, df in [('Train', train_df), ('Test', test_df)]:
        # 가중 분담률: choice_prob * n_total
        df_tmp = df.copy()
        df_tmp['weighted'] = df_tmp['choice_prob'] * df_tmp['n_total']
        share = df_tmp.groupby('transport_category')['weighted'].sum()
        share = (share / share.sum() * 100).sort_values(ascending=False)
        n_ods = df['od_pair'].nunique()
        print(f'\n  {label} mode share ({n_ods:,} ODs):')
        for cat, pct in share.items():
            print(f'    {cat:<20s} {pct:5.1f}%')


def create_dataloaders(data_dir='../../../data/choice_set/java_default', batch_size=2048,
                       num_workers=0, device='cpu', features=None):
    """Train/Test DataLoader 생성."""
    train_df, test_df = load_and_split(data_dir)

    print_mode_share(train_df, test_df)

    features = features or MODEL_FEATURES
    train_ds = RouteChoiceDataset(train_df, fit_scaler=True, features=features)
    test_ds = RouteChoiceDataset(
        test_df,
        scaler=train_ds.scaler,
        context_scaler=train_ds.context_scaler,
        features=features,
    )

    use_pin_memory = (device != 'cpu' and torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    print(f'\n  Train: {len(train_ds):,} ODs, Test: {len(test_ds):,} ODs')
    feat_names = [f for f in features]
    print(f'  Features ({len(features)}): {", ".join(feat_names)}')
    print(f'  Batch size: {batch_size}')

    return train_loader, test_loader, train_ds, test_ds
