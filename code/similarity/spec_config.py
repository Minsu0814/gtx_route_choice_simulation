"""
Shared spec definitions for 4-Spec × 3-Model comparison.
wait_time_min removed, ASC dummies used instead of has_bus/train/gtx.

Specs:
  A: Total IVT, no walk
  B: Total IVT + walk
  C: IVT split, no walk
  D: IVT split + walk

Models: MNL (con+uncon), Mixed Logit, TasteNet
"""

CATEGORY_MAP = {
    'bus_only': 'bus_only', 'train_only': 'train_only',
    'bus+train': 'bus+train', 'train+gtx': 'train+gtx',
    'gtx_only': 'train+gtx', 'bus+train+gtx': 'train+gtx',
    'bus+gtx': 'train+gtx',
}
ASC_REF = 'bus_only'

# Common base features (no wait_time_min)
_BASE = ['transfer_walk_time_min', 'num_transfers', 'fare_1000won']
_WALK = ['ln_access', 'ln_egress']
_IVT_TOTAL = ['total_ivt_min']
_IVT_SPLIT = ['ivt_bus', 'ivt_train', 'ivt_gtx']
# ASC dummies for TasteNet (replaces has_bus/train/gtx)
_ASC_DUMMIES = ['is_train_only', 'is_bus_train', 'is_train_gtx']

# --- MNL specs (8 = 4 specs × con/uncon) ---
_ALL_NEG = set()  # unconstrained

MNL_SPECS = {
    'A_total_con': {
        'features': _IVT_TOTAL + _BASE,
        'sign_neg': set(_IVT_TOTAL + _BASE),
    },
    'A_total_uncon': {
        'features': _IVT_TOTAL + _BASE,
        'sign_neg': _ALL_NEG,
    },
    'B_total_walk_con': {
        'features': _IVT_TOTAL + _BASE + _WALK,
        'sign_neg': set(_IVT_TOTAL + _BASE + _WALK),
    },
    'B_total_walk_uncon': {
        'features': _IVT_TOTAL + _BASE + _WALK,
        'sign_neg': _ALL_NEG,
    },
    'C_split_con': {
        'features': _IVT_SPLIT + _BASE,
        'sign_neg': set(_IVT_SPLIT + _BASE),
    },
    'C_split_uncon': {
        'features': _IVT_SPLIT + _BASE,
        'sign_neg': _ALL_NEG,
    },
    'D_split_walk_con': {
        'features': _IVT_SPLIT + _BASE + _WALK,
        'sign_neg': set(_IVT_SPLIT + _BASE + _WALK),
    },
    'D_split_walk_uncon': {
        'features': _IVT_SPLIT + _BASE + _WALK,
        'sign_neg': _ALL_NEG,
    },
}

# --- Mixed Logit specs (4) ---
ML_SPECS = {
    'ML_A_total': {
        'random_ln': ['total_ivt_min'],
        'random_normal': ['transfer_walk_time_min', 'num_transfers'],
        'fixed': ['fare_1000won'],
    },
    'ML_B_total_walk': {
        'random_ln': ['total_ivt_min'],
        'random_normal': ['transfer_walk_time_min', 'num_transfers'],
        'fixed': ['fare_1000won', 'ln_access', 'ln_egress'],
    },
    'ML_C_split': {
        'random_ln': ['ivt_bus', 'ivt_train', 'ivt_gtx'],
        'random_normal': ['transfer_walk_time_min', 'num_transfers'],
        'fixed': ['fare_1000won'],
    },
    'ML_D_split_walk': {
        'random_ln': ['ivt_bus', 'ivt_train', 'ivt_gtx'],
        'random_normal': ['transfer_walk_time_min', 'num_transfers'],
        'fixed': ['fare_1000won', 'ln_access', 'ln_egress'],
    },
}

# --- TasteNet specs (4) — uses ASC dummies as features ---
TN_SPECS = {
    'TN_A_total': _IVT_TOTAL + _BASE + _ASC_DUMMIES,
    'TN_B_total_walk': _IVT_TOTAL + _BASE + _WALK + _ASC_DUMMIES,
    'TN_C_split': _IVT_SPLIT + _BASE + _ASC_DUMMIES,
    'TN_D_split_walk': _IVT_SPLIT + _BASE + _WALK + _ASC_DUMMIES,
}

CONTEXT_FEATURES = ['od_distance_km', 'choice_set_size']
