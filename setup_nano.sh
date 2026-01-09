#!/usr/bin/env bash
set -euo pipefail

# Jetson Nano (JetPack 4.x) inference environment setup.
# Goal: Python 3.6 + prebuilt PyTorch wheel + stable-baselines3 compatible versions.
# This script avoids any simulator dependencies.

ENV_DIR="${ENV_DIR:-.venv_nano_py36}"
PYTHON_BIN="${PYTHON_BIN:-python3.6}"

echo "[setup_nano] Using python: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. On Jetson Nano (Ubuntu 18.04), install python3.6 first." >&2
  exit 1
fi

# Use pip-installed virtualenv to avoid apt python3.6-venv version mismatch
if [[ ! -d "${ENV_DIR}" ]]; then
  echo "[setup_nano] Installing virtualenv via pip (avoids apt version conflicts)"
  "${PYTHON_BIN}" -m pip install --user virtualenv
  echo "[setup_nano] Creating virtualenv at ${ENV_DIR} (with system site-packages so cv2 from apt works)"
  "${PYTHON_BIN}" -m virtualenv --system-site-packages "${ENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

echo "[setup_nano] Upgrading pip tooling (Python 3.6 compatible pins)"
python -m pip install --upgrade "pip==21.3.1" "setuptools<65" wheel

# Use system numpy (from apt) to avoid Illegal Instruction on Nano's Cortex-A57.
# Do NOT pip install numpy; rely on --system-site-packages.
echo "[setup_nano] Checking system numpy (from apt)..."
if ! python -c "import numpy; print('numpy ok:', numpy.__version__)" 2>/dev/null; then
  echo "ERROR: System numpy not working. Install via apt:" >&2
  echo "  sudo apt-get install -y python3-numpy" >&2
  exit 1
fi

echo "[setup_nano] Installing pillow"
python -m pip install --upgrade pillow

# OpenCV: prefer JetPack apt packages, accessed via --system-site-packages.
python - <<'PY'
try:
    import cv2
    print('[setup_nano] cv2 OK:', cv2.__version__)
except Exception as e:
    print('[setup_nano] cv2 not available in this environment.')
    print('Install on Jetson: sudo apt-get update && sudo apt-get install -y python3-opencv')
    raise SystemExit(0)
PY

# PyTorch: user supplies a prebuilt wheel path/URL.
# Example override:
#   export TORCH_WHEEL=/home/jetson/Downloads/torch-*-cp36-cp36m-linux_aarch64.whl
#   bash setup_nano.sh
#
# Default wheel URL (user provided):
DEFAULT_TORCH_WHEEL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"

TORCH_WHEEL_TO_USE="${TORCH_WHEEL:-$DEFAULT_TORCH_WHEEL}"

echo "[setup_nano] Installing PyTorch wheel: ${TORCH_WHEEL_TO_USE}"
python -m pip install --no-cache-dir "${TORCH_WHEEL_TO_USE}"

# SB3 for Python 3.6: use a version that still supports py3.6.
# gym 0.21 is commonly used with SB3 1.2.x and py3.6.
echo "[setup_nano] Installing SB3 (Python 3.6 compatible)"
python -m pip install --upgrade "gym==0.21.0" "stable-baselines3==1.2.0" cloudpickle

echo "[setup_nano] Done. Activate with: source ${ENV_DIR}/bin/activate"
