#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Jetson Nano (JetPack 4.x) Inference Environment Setup
# =============================================================================
# This script creates a Python 3.6 virtualenv with:
#   - System numpy/opencv (from apt, avoids Illegal Instruction)
#   - PyTorch (prebuilt NVIDIA wheel)
#   - stable-baselines3 1.2.0 + gym 0.21.0
#
# Prerequisites (run these BEFORE this script):
#   sudo apt-get update
#   sudo apt-get install -y python3-pip python3-numpy python3-opencv
#
# Usage:
#   bash setup_nano.sh
#
# To use a different torch wheel:
#   export TORCH_WHEEL=/path/to/torch.whl
#   bash setup_nano.sh
# =============================================================================

ENV_DIR="${ENV_DIR:-.venv_nano_py36}"
PYTHON_BIN="${PYTHON_BIN:-python3.6}"

# Default PyTorch wheel for JetPack 4.x (user-provided URL)
DEFAULT_TORCH_WHEEL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
TORCH_WHEEL_TO_USE="${TORCH_WHEEL:-$DEFAULT_TORCH_WHEEL}"

echo "============================================================"
echo "[setup_nano] Jetson Nano Inference Environment Setup"
echo "============================================================"
echo "[setup_nano] Python:     ${PYTHON_BIN}"
echo "[setup_nano] Venv dir:   ${ENV_DIR}"
echo "[setup_nano] Torch URL:  ${TORCH_WHEEL_TO_USE}"
echo "============================================================"

# --- Step 1: Check python3.6 exists ---
echo ""
echo "[Step 1/8] Checking Python 3.6..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 1
fi
echo "  OK: $(${PYTHON_BIN} --version)"

# --- Step 2: Check system numpy (apt) ---
echo ""
echo "[Step 2/8] Checking system numpy (from apt)..."
if ! "${PYTHON_BIN}" -c "import numpy; print('  OK: numpy', numpy.__version__)" 2>/dev/null; then
  echo "ERROR: System numpy not available. Run:" >&2
  echo "  sudo apt-get install -y python3-numpy" >&2
  exit 1
fi

# --- Step 3: Check system opencv (apt) ---
echo ""
echo "[Step 3/8] Checking system OpenCV (from apt)..."
if ! "${PYTHON_BIN}" -c "import cv2; print('  OK: cv2', cv2.__version__)" 2>/dev/null; then
  echo "ERROR: System OpenCV not available. Run:" >&2
  echo "  sudo apt-get install -y python3-opencv" >&2
  exit 1
fi

# --- Step 4: Install virtualenv ---
echo ""
echo "[Step 4/8] Installing virtualenv..."
"${PYTHON_BIN}" -m pip install --user --quiet virtualenv
echo "  OK"

# --- Step 5: Create venv (or reuse existing) ---
echo ""
echo "[Step 5/8] Creating virtualenv..."
if [[ -d "${ENV_DIR}" ]]; then
  echo "  Venv already exists at ${ENV_DIR}, reusing."
else
  "${PYTHON_BIN}" -m virtualenv --system-site-packages "${ENV_DIR}"
  echo "  Created: ${ENV_DIR}"
fi

# --- Activate venv ---
# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"
echo "  Activated venv. Python: $(which python)"

# --- Step 6: Upgrade pip/setuptools ---
echo ""
echo "[Step 6/8] Upgrading pip, setuptools, wheel..."
python -m pip install --quiet --upgrade "pip==21.3.1" "setuptools<65" wheel
echo "  OK: pip $(pip --version | awk '{print $2}')"

# --- Step 7: Install PyTorch ---
echo ""
echo "[Step 7/8] Installing PyTorch..."

# We use a fixed local filename for the cached wheel
# This corresponds to the default URL's content (PyTorch 1.10.0)
CACHED_WHEEL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"

if [[ "${TORCH_WHEEL_TO_USE}" =~ ^https?:// ]]; then
  echo "  Source is URL: ${TORCH_WHEEL_TO_USE}"
  
  # Check if the cached wheel already exists
  if [[ -f "${CACHED_WHEEL}" ]]; then
    echo "  Found local file '${CACHED_WHEEL}'."
    # Verify it acts like a zip file (wheels are zip files)
    if ! python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1])" "${CACHED_WHEEL}" >/dev/null 2>&1; then
      echo "  WARNING: Local file is corrupt or not a zip file. Deleting..."
      rm -f "${CACHED_WHEEL}"
    else
      echo "  Local file verified as valid."
    fi
  fi

  if [[ -f "${CACHED_WHEEL}" ]]; then
    echo "  Skipping download, using existing '${CACHED_WHEEL}'."
    WHEEL_TO_INSTALL="${CACHED_WHEEL}"
  else
    echo "  Downloading to '${CACHED_WHEEL}'..."
    if command -v wget >/dev/null 2>&1; then
      wget --quiet --show-progress --no-check-certificate -O "${CACHED_WHEEL}" "${TORCH_WHEEL_TO_USE}"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -o "${CACHED_WHEEL}" "${TORCH_WHEEL_TO_USE}"
    else
      echo "ERROR: wget/curl not found, cannot download wheel."
      exit 1
    fi
    WHEEL_TO_INSTALL="${CACHED_WHEEL}"
  fi
  
  echo "  Installing ${WHEEL_TO_INSTALL}..."
  python -m pip install --quiet --no-cache-dir "${WHEEL_TO_INSTALL}"
  # We do NOT remove the file, so it can be reused
else
  echo "  Source is local file: ${TORCH_WHEEL_TO_USE}"
  python -m pip install --quiet --no-cache-dir "${TORCH_WHEEL_TO_USE}"
fi

if python -c "import torch; print('  OK: torch', torch.__version__)" 2>/dev/null; then
  :
else
  echo "WARNING: torch import failed. May be wheel mismatch with your JetPack version."
fi

# --- Step 8: Install SB3 + gym ---
echo ""
echo "[Step 8/8] Installing stable-baselines3, gym, cloudpickle, pillow..."
python -m pip install --quiet --upgrade \
  "gym==0.21.0" \
  "stable-baselines3==1.2.0" \
  cloudpickle \
  pickle5 \
  gymnasium \
  pillow \
  onnx
  onnxruntime

# Verify SB3
if python -c "import stable_baselines3; print('  OK: stable-baselines3', stable_baselines3.__version__)" 2>/dev/null; then
  :
else
  echo "ERROR: stable-baselines3 import failed!" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "[setup_nano] SUCCESS!"
echo "============================================================"
echo ""
echo "To activate this environment:"
echo "  source ${ENV_DIR}/bin/activate"
echo ""
echo "To run inference (mock mode):"
echo "  python run_policy_real.py --mode mock --model models/<run>/best_model.zip --images-dir data/road --fps 15"
echo ""
