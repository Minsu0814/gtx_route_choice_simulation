"""
Shared feature transformations for MNL training and assignment.
Ensures identical preprocessing in both pipelines.
"""
import math
import numpy as np

from spec_config import CATEGORY_MAP, ASC_REF


def transform_raw_features(feat: dict) -> dict:
    """Transform raw alt_features (seconds/won) to model units.

    Converts:
      - time fields: seconds → minutes
      - fare: won → 1000won
      - total_ivt_min: in_vehicle_time / 60
      - ln_access, ln_egress: log(1 + minutes)
      - transport_category: mapped via CATEGORY_MAP
    """
    out = dict(feat)
    # seconds -> minutes
    for col in ['ivt_bus', 'ivt_train', 'ivt_gtx',
                'waiting_time', 'transfer_walk_time',
                'in_vehicle_time', 'access_time', 'egress_time']:
        if col in out and out[col] is not None:
            out[col] = float(out[col]) / 60.0

    out['transfer_walk_time_min'] = out.get('transfer_walk_time', 0.0) or 0.0
    out['fare_1000won'] = (float(out.get('fare', 0) or 0)) / 1000.0
    out['total_ivt_min'] = out.get('in_vehicle_time', 0.0) or 0.0
    out['ln_access'] = math.log1p(out.get('access_time', 0.0) or 0.0)
    out['ln_egress'] = math.log1p(out.get('egress_time', 0.0) or 0.0)

    raw_cat = out.get('transport_category', '')
    out['transport_category_mapped'] = CATEGORY_MAP.get(raw_cat, raw_cat)
    return out


def build_feature_vector(feat: dict, feature_names: list,
                         asc_cats: list) -> np.ndarray:
    """Build numpy feature vector matching beta order.

    Args:
        feat: transformed feature dict (from transform_raw_features)
        feature_names: spec feature names (e.g. ['total_ivt_min', ...])
        asc_cats: sorted ASC category names (excluding ASC_REF)

    Returns:
        1-D array of length len(feature_names) + len(asc_cats)
    """
    vals = [float(feat.get(f, 0.0) or 0.0) for f in feature_names]
    cat = feat.get('transport_category_mapped', '')
    for c in asc_cats:
        vals.append(1.0 if cat == c else 0.0)
    return np.array(vals, dtype=np.float64)
