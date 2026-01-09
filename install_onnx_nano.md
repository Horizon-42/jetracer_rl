# Jetson Nano (JetPack 4.6) ONNX & ONNX Runtime Installation Guide

**Environment:**
* **Device:** NVIDIA Jetson Nano
* **OS:** Ubuntu 18.04 (JetPack 4.6)
* **Python:** 3.6
* **Date:** 2025

This document outlines the step-by-step process to successfully install ONNX and ONNX Runtime-GPU. It specifically addresses the "dependency hell" issue with `protobuf-compiler` on Ubuntu 18.04.

---

## 1. Fix System Dependencies (Critical Step)

On Jetson Nano (Ubuntu 18.04), `apt` often fails to install `protobuf-compiler` because of a version mismatch between `libprotoc10` (which defaults to a newer version) and the compiler itself.

**Error Message:**
> *Depends: libprotoc10 (= 3.0.0-9.1ubuntu1) but 3.0.0-9.1ubuntu1.1 is to be installed*

**Solution:**
Force-install the specific matching versions for all protobuf-related packages.

```bash
sudo apt-get update

# Force install specific versions to resolve "held broken packages"
sudo apt-get install -y \
    libprotoc10=3.0.0-9.1ubuntu1 \
    protobuf-compiler=3.0.0-9.1ubuntu1 \
    libprotobuf-dev=3.0.0-9.1ubuntu1 \
    libprotoc-dev=3.0.0-9.1ubuntu1

# Install remaining system tools
sudo apt-get install -y python3-pip cmake python3-dev
```

## 2. Python Environment Setup
Python 3.6 is End-of-Life, so we must pin specific versions for compatibility.

```bash

# Upgrade pip to ensure it can handle wheel files correctly
python3 -m pip install --upgrade pip

# Install build dependencies
python3 -m pip install Cython

# Install Protobuf for Python
# Note: 3.19.6 is the last version supporting Python 3.6
python3 -m pip install protobuf==3.19.6
```

## 3. Install ONNX Library
Install the standard ONNX library for model definitions.

```bash
# Version 1.11.0 is compatible with Py3.6 and the protobuf version above
python3 -m pip install onnx==1.11.0
```

## 4. Install ONNX Runtime GPU
You cannot install onnxruntime-gpu directly from PyPI on Jetson. You must use the NVIDIA-provided wheel file.

Target Version: 1.10.0 (Compatible with CUDA 10.2 in JetPack 4.6)

```bash
# 1. Download the pre-built wheel from NVIDIA
Find suitable wheel from https://elinux.org/Jetson_Zoo#ONNX_Runtime

# 2. Install the wheel
python3 -m pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

## 5. Verification
Run the following Python script to confirm installation and GPU detection.

```bash
python3 -c "
import onnx
import onnxruntime as ort

print('--------------------------------')
print(f'ONNX Version: {onnx.__version__}')
print(f'ORT Device:   {ort.get_device()}')
print(f'Providers:    {ort.get_available_providers()}')
print('--------------------------------')
"
```

Expected Output:

ONNX Version: 1.11.0

ORT Device: GPU

Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']

### Troubleshooting Notes
- Illegal Instruction (Core Dumped): This often happens if numpy was upgraded to a version not optimized for ARM64. Fix: pip install numpy==1.19.5

- CMake Error during install: Ensure you ran sudo apt-get install cmake.

- No matching distribution for protobuf: Double-check you typed ==3.19.6 correctly.