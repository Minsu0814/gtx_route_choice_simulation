"""
Run Raptor GC parameter calibration.

Usage:
    python run_raptor_calibration.py
    python run_raptor_calibration.py --spec A_total_uncon
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code' / 'similarity' / 'analysis'))

from raptor_calibration_prep import run_calibration

DATA_DIR = ROOT / 'data' / 'results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='A_total_uncon')
    args = parser.parse_args()

    assign_path = DATA_DIR / f'assignment_{args.spec}.parquet'
    train_path = DATA_DIR / 'route_choice_filtered_training.parquet'

    results = run_calibration(assign_path, train_path)

    out_path = DATA_DIR / 'raptor_calibration_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved: {out_path}')


if __name__ == '__main__':
    main()
