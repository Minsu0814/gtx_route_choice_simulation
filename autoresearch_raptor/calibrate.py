"""
Raptor Calibration - Default vs Calibrated Comparison.

Usage: python calibrate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import run_raptor, log_result, print_comparison  # noqa: E402

# ============================================================
# PARAMS
# ============================================================
DEFAULT_PARAMS = {
    'transfer_cost_secs': 120,
    'walk_reluctance': 1.0,
    'bus_reluctance': 1.0,
    'train_reluctance': 1.0,
    'wait_reluctance': 1.0,
}

CALIBRATED_PARAMS = {
    'transfer_cost_secs': 600,
    'walk_reluctance': 10.0,
    'bus_reluctance': 0.7,
    'train_reluctance': 1.5,
    'wait_reluctance': 1.5,
}

N_SAMPLE = 2000
# ============================================================


def main():
    print('=== Raptor Calibration: Default vs Calibrated ===', flush=True)
    print(f'  Sample: {N_SAMPLE} ODs\n', flush=True)

    print('--- [1/2] Default params ---', flush=True)
    print(f'  {DEFAULT_PARAMS}', flush=True)
    m_default = run_raptor(DEFAULT_PARAMS, n_sample=N_SAMPLE)

    print('\n--- [2/2] Calibrated params ---', flush=True)
    print(f'  {CALIBRATED_PARAMS}', flush=True)
    m_calib = run_raptor(CALIBRATED_PARAMS, n_sample=N_SAMPLE)

    print('\n=== COMPARISON ===', flush=True)
    print_comparison(m_default, m_calib)

    log_result('default', DEFAULT_PARAMS, m_default)
    log_result('calibrated', CALIBRATED_PARAMS, m_calib)
    print(f'\n  Logged to results.tsv', flush=True)


if __name__ == '__main__':
    main()
