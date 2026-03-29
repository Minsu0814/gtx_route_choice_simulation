"""
Deep Learning Route Choice — 모델 정의.

1. DNNChoiceModel:  Feedforward → utility → masked softmax
2. TasteNetModel:   β = g(context) → V = Xβ(z)  (해석 가능)
3. ResLogitModel:   V = Xβ_MNL + DNN(X)  (MNL 잔차 학습)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path


def masked_softmax(logits, mask):
    """패딩된 대안을 -inf로 마스킹한 softmax.

    Args:
        logits: (batch, MAX_ALTS)
        mask:   (batch, MAX_ALTS) — 1=valid, 0=pad
    Returns:
        probs:  (batch, MAX_ALTS) — 확률 (패딩=0)
    """
    logits = logits.masked_fill(mask == 0, -1e9)
    return F.softmax(logits, dim=-1) * mask


class DNNChoiceModel(nn.Module):
    """DNN: 각 대안의 피처 → utility score → masked softmax.

    Architecture: Linear(n_feat→64) → ReLU → Dropout
                → Linear(64→32) → ReLU → Dropout
                → Linear(32→1) → utility
    """

    def __init__(self, n_features, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X, z, mask):
        """
        Args:
            X:    (batch, MAX_ALTS, n_features)
            z:    (batch, n_context) — unused in DNN
            mask: (batch, MAX_ALTS)
        Returns:
            probs: (batch, MAX_ALTS)
        """
        # X: (B, A, F) → utility per alt
        V = self.net(X).squeeze(-1)  # (B, A)
        return masked_softmax(V, mask)


class TasteNetModel(nn.Module):
    """TasteNet: context → β(z), V = X @ β(z).

    β_i = g(z) where z = context features (od_distance_km, choice_set_size).
    해석: OD 거리에 따라 시간/비용 감수성이 달라짐.

    Architecture:
        taste_net: Linear(n_context→32) → ReLU → Linear(32→n_features)
        V = sum_f X_f * β_f(z)
    """

    def __init__(self, n_features, n_context, hidden_dim=32):
        super().__init__()
        self.n_features = n_features
        self.taste_net = nn.Sequential(
            nn.Linear(n_context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, X, z, mask):
        """
        Args:
            X:    (batch, MAX_ALTS, n_features)
            z:    (batch, n_context)
            mask: (batch, MAX_ALTS)
        Returns:
            probs: (batch, MAX_ALTS)
        """
        beta = self.taste_net(z)          # (B, F)
        V = (X * beta.unsqueeze(1)).sum(-1)  # (B, A)
        return masked_softmax(V, mask)

    def get_betas(self, z):
        """Context별 β 값 반환 (해석용)."""
        with torch.no_grad():
            return self.taste_net(z)


class ResLogitModel(nn.Module):
    """ResLogit: V = X @ β_MNL + DNN_residual(X).

    MNL의 선형 유틸리티 위에 비선형 잔차를 학습.
    β_MNL은 사전 추정값으로 초기화 (고정 또는 미세조정 가능).

    Architecture:
        linear: V_MNL = X @ β  (MNL 초기화)
        residual_net: Linear(F→32) → ReLU → Dropout → Linear(32→1)
        V = V_MNL + residual_net(X)
    """

    def __init__(self, n_features, mnl_beta=None,
                 hidden_dim=32, dropout=0.1, freeze_mnl=False):
        super().__init__()
        self.n_features = n_features
        self.freeze_mnl = freeze_mnl

        # Linear part (MNL)
        self.beta_mnl = nn.Parameter(torch.zeros(n_features))
        if mnl_beta is not None:
            self.beta_mnl.data = torch.tensor(mnl_beta, dtype=torch.float32)
        if freeze_mnl:
            self.beta_mnl.requires_grad = False

        # Residual DNN
        self.residual_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize residual near zero
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, X, z, mask):
        """
        Args:
            X:    (batch, MAX_ALTS, n_features)
            z:    (batch, n_context) — unused
            mask: (batch, MAX_ALTS)
        Returns:
            probs: (batch, MAX_ALTS)
        """
        V_mnl = (X * self.beta_mnl).sum(-1)     # (B, A)
        V_res = self.residual_net(X).squeeze(-1)  # (B, A)
        V = V_mnl + V_res
        return masked_softmax(V, mask)

    def get_residual_ratio(self, X, mask):
        """잔차 비중 분석 (해석용)."""
        with torch.no_grad():
            V_mnl = (X * self.beta_mnl).sum(-1)
            V_res = self.residual_net(X).squeeze(-1)
            valid = mask.bool()
            mnl_mag = V_mnl[valid].abs().mean().item()
            res_mag = V_res[valid].abs().mean().item()
            total = mnl_mag + res_mag + 1e-8
            return {
                'mnl_magnitude': mnl_mag,
                'residual_magnitude': res_mag,
                'residual_ratio': res_mag / total,
            }


class ASUDNNModel(nn.Module):
    """ASU-DNN: Alternative-Specific Utility DNN.

    대안별 독립 sub-network로 utility 학습.
    각 대안이 고유한 가중치를 가져 대안 간 이질성 포착.
    Han et al. (2020), "A neural-embedded discrete choice model"

    Architecture (per alternative):
        Linear(n_feat→hidden) → ReLU → Dropout → Linear(hidden→1) → utility
    """

    def __init__(self, n_features, max_alts=5, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.max_alts = max_alts
        self.alt_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(max_alts)
        ])

    def forward(self, X, z, mask):
        """
        Args:
            X:    (batch, MAX_ALTS, n_features)
            z:    (batch, n_context) — unused
            mask: (batch, MAX_ALTS)
        Returns:
            probs: (batch, MAX_ALTS)
        """
        B = X.size(0)
        V = torch.zeros(B, self.max_alts, device=X.device)
        for j, net in enumerate(self.alt_nets):
            V[:, j] = net(X[:, j, :]).squeeze(-1)
        return masked_softmax(V, mask)


class LMNLModel(nn.Module):
    """L-MNL: Learning MNL.

    DNN이 raw features → latent variables 생성,
    이를 MNL의 추가 설명변수로 사용.
    Sifringer et al. (2020), "Enhancing discrete choice models with representation learning"

    Architecture:
        feature_extractor: Linear(F→32) → ReLU → Linear(32→n_latent)
        V = X_aug @ beta   where X_aug = [X_original, latent_features]
    """

    def __init__(self, n_features, n_latent=4, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_latent = n_latent

        # DNN feature extractor: raw features → latent variables
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_latent),
        )

        # Linear utility: original + latent features → scalar
        self.beta = nn.Linear(n_features + n_latent, 1, bias=False)

    def forward(self, X, z, mask):
        """
        Args:
            X:    (batch, MAX_ALTS, n_features)
            z:    (batch, n_context) — unused
            mask: (batch, MAX_ALTS)
        Returns:
            probs: (batch, MAX_ALTS)
        """
        latent = self.feature_extractor(X)          # (B, A, n_latent)
        X_aug = torch.cat([X, latent], dim=-1)      # (B, A, F+n_latent)
        V = self.beta(X_aug).squeeze(-1)            # (B, A)
        return masked_softmax(V, mask)

    def get_betas(self):
        """Linear utility 가중치 반환 (해석용)."""
        with torch.no_grad():
            return self.beta.weight.squeeze(0).cpu().numpy()


def load_mnl_beta(coeff_path, scaler, features=None):
    """MNL β를 StandardScaler 공간으로 변환.

    원본 β는 raw feature 공간: V = β·x
    Scaled 공간: x' = (x - μ) / σ, V = β·(σ·x' + μ) = (β·σ)·x' + β·μ
    → scaled β = β_raw * σ  (상수항은 softmax에서 상쇄)
    """
    from .data import MODEL_FEATURES
    features = features or MODEL_FEATURES

    with open(coeff_path, 'r', encoding='utf-8') as f:
        coeff = json.load(f)

    beta_raw = np.array([coeff['beta'][f] for f in features], dtype=np.float32)
    sigma = scaler.scale_.astype(np.float32)
    beta_scaled = beta_raw * sigma

    return beta_scaled
