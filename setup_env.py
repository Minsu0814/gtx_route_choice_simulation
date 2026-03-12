"""
환경 설치 스크립트 — CUDA 자동 감지 + PyTorch + 의존성 설치.

Usage:
    python setup_env.py          # 자동 CUDA 감지
    python setup_env.py --cpu    # CPU 전용 강제
"""

import subprocess
import sys
import re
import argparse


def detect_cuda_version():
    """nvidia-smi로 CUDA 버전 감지. 실패 시 None 반환."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        match = re.search(r'CUDA Version:\s+([\d.]+)', result.stdout)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_torch_install_args(cuda_version):
    """CUDA 버전에 맞는 PyTorch 설치 인자 반환."""
    base_packages = ['torch', 'torchvision', 'torchaudio']

    if cuda_version is None:
        # CPU only
        return base_packages + [
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]

    major_minor = tuple(int(x) for x in cuda_version.split('.')[:2])

    if major_minor >= (12, 4):
        suffix = 'cu124'
    elif major_minor >= (12, 1):
        suffix = 'cu121'
    elif major_minor >= (11, 8):
        suffix = 'cu118'
    else:
        print(f'  [WARN] CUDA {cuda_version} is old — falling back to CPU build')
        return base_packages + [
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]

    return base_packages + [
        '--index-url', f'https://download.pytorch.org/whl/{suffix}'
    ]


def pip_install(args):
    """pip install 실행."""
    cmd = [sys.executable, '-m', 'pip', 'install'] + args
    print(f'  $ {" ".join(cmd)}')
    subprocess.check_call(cmd)


def verify_installation():
    """설치 검증."""
    print('\n=== Installation Verification ===')
    errors = []

    # torch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
        print(f'  torch {torch.__version__} — CUDA: {cuda_available} ({device_name})')
    except ImportError:
        errors.append('torch')
        print('  [FAIL] torch not found')

    # pandas
    try:
        import pandas as pd
        print(f'  pandas {pd.__version__}')
    except ImportError:
        errors.append('pandas')
        print('  [FAIL] pandas not found')

    # sklearn
    try:
        import sklearn
        print(f'  scikit-learn {sklearn.__version__}')
    except ImportError:
        errors.append('scikit-learn')
        print('  [FAIL] scikit-learn not found')

    # numpy
    try:
        import numpy as np
        print(f'  numpy {np.__version__}')
    except ImportError:
        errors.append('numpy')
        print('  [FAIL] numpy not found')

    if errors:
        print(f'\n  [ERROR] Failed packages: {", ".join(errors)}')
        return False
    else:
        print('\n  All packages installed successfully!')
        return True


def main():
    parser = argparse.ArgumentParser(description='Setup environment for DL route choice')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only PyTorch')
    args = parser.parse_args()

    print('=== Deep Learning Route Choice — Environment Setup ===\n')

    # Step 1: Detect CUDA
    if args.cpu:
        print('[1/3] CUDA detection skipped (--cpu flag)')
        cuda_version = None
    else:
        print('[1/3] Detecting CUDA version...')
        cuda_version = detect_cuda_version()
        if cuda_version:
            print(f'  Found CUDA {cuda_version}')
        else:
            print('  No CUDA detected — installing CPU-only PyTorch')

    # Step 2: Install PyTorch
    print('\n[2/3] Installing PyTorch...')
    torch_args = get_torch_install_args(cuda_version)
    pip_install(torch_args)

    # Step 3: Install other dependencies
    print('\n[3/3] Installing project dependencies...')
    pip_install(['-r', 'requirements.txt'])

    # Verify
    success = verify_installation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
