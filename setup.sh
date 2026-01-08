# 1. 先降级 setuptools (这是安装 gym 0.21 的唯一方法)
pip install "setuptools<65" wheel

# 2. 安装其余依赖
pip install -r requirements.txt

# 3. 安装 DonkeyCar Python 库
pip install git+https://github.com/tawnkramer/gym-donkeycar.git#egg=gym-donkeycar

# 4. (可选) 安装 GPU 版 PyTorch (根据你的 CUDA 版本)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


