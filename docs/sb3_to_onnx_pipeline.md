# SB3 to ONNX Model Conversion Pipeline

本文档简述从 Stable-Baselines3 (SB3) 模型到 ONNX 模型的转换流程及注意事项。

---

## 概述

将训练好的 SB3 PPO 模型转换为 ONNX 格式，以便在 Jetson Nano 等边缘设备上进行高效推理。转换过程包括：

1. **模型包装**：提取策略网络（Actor），去除 Value Net 和 Log Prob 计算
2. **ONNX 导出**：使用 PyTorch 的 ONNX 导出功能
3. **验证**：对比 PyTorch 和 ONNX Runtime 的输出一致性
4. **部署**：在目标设备上使用 ONNX Runtime 进行推理

---

## 转换流程

### 1. 模型包装 (`OnnxablePolicy`)

SB3 的策略网络包含 Actor、Critic 等多个组件。转换时只需要 Actor 部分（策略网络），并且需要输出确定性动作（均值）。

```python
class OnnxablePolicy(th.nn.Module):
    """包装 SB3 的策略，只输出确定性的动作 (Deterministic Action / Mean)"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # deterministic=True 返回高斯分布的均值
        return self.policy(observation, deterministic=True)[0]
```

**关键点：**
- 使用 `deterministic=True` 确保输出是动作分布的均值，而非采样值
- 只返回 `[0]`（动作），忽略 `values` 和 `log_probs`

### 2. 模型加载与导出

```python
# 加载 SB3 模型（强制使用 CPU）
model = PPO.load(model_path, device="cpu")

# 创建包装后的模型
onnx_policy = OnnxablePolicy(model.policy)

# 创建 Dummy Input（匹配观察空间形状）
dummy_input = th.randn(1, channels, height, width)

# 导出为 ONNX
th.onnx.export(
    onnx_policy,
    dummy_input,
    output_path,
    opset_version=11,          # Jetson/Py3.6 建议使用 opset 11
    input_names=["input"],
    output_names=["action"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "action": {0: "batch_size"}
    }
)
```

**关键参数：**
- `opset_version=11`：兼容 Jetson Nano (Python 3.6) 和 ONNX Runtime 1.10.0
- `dynamic_axes`：支持动态 batch size（推理时可以使用不同的 batch size）

### 3. 验证导出结果

```python
# 1. PyTorch 推理
with th.no_grad():
    torch_out = torch_model(dummy_input).numpy()

# 2. ONNX Runtime 推理
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

# 3. 对比输出（允许一定的浮点误差）
np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
```

**验证标准：**
- 相对误差 `rtol=1e-03`（0.1%）
- 绝对误差 `atol=1e-05`
- 如果验证失败，检查模型结构和输入格式

---

## 使用脚本

项目提供了转换脚本 `export_sb3_ppo_to_onnx.py`：

```bash
python export_sb3_ppo_to_onnx.py \
    --model models/your_model/last_model.zip \
    --output onnx_models \
    --height 84 \
    --width 84 \
    --channels 3
```

**参数说明：**
- `--model`：SB3 模型路径（.zip 文件）
- `--output`：输出文件夹（默认：onnx_models）
- `--height/--width/--channels`：观察空间维度（需与训练时一致）

---

## 注意事项

### 1. 设备兼容性

- **训练环境**：使用 CPU 加载模型（`device="cpu"`），避免 CUDA 设备不匹配
- **Jetson Nano**：使用 `opset_version=11` 确保与 ONNX Runtime 1.10.0 兼容
- **Python 版本**：训练环境通常使用 Python 3.9+，Jetson Nano 使用 Python 3.6

### 2. 输入格式一致性

**训练时的预处理：**
- 图像格式：RGB，HWC → CHW（channels-first）
- 归一化：`[0, 255]` → `[0.0, 1.0]`（float32）
- 形状：`(batch, channels, height, width)`

**推理时的预处理必须完全一致：**
```python
# 1. 图像预处理（透视变换、resize 等）
# 2. BGR → RGB
# 3. HWC → CHW
# 4. 归一化到 [0, 1]
# 5. 添加 batch 维度
obs = preprocess_image(frame, model_width, model_height, ...)
```

### 3. 观察模式匹配

确保推理时的观察模式（`obs_mode`）与训练时一致：
- `raw`：原始图像
- `perspective`：鸟瞰图
- `mix`：原始 + 鸟瞰图垂直堆叠
- `mask`：HSV 颜色阈值提取的掩码

**检查点：**
- 透视变换矩阵参数是否一致
- 图像尺寸是否匹配（训练时 84x84，推理时也应该是 84x84）
- 颜色空间转换是否正确（BGR → RGB）

### 4. 输出处理

ONNX 模型输出的是原始动作值，可能需要后处理：

```python
# ONNX 推理
action_raw = sess.run(None, {input_name: obs})[0].flatten()

# 后处理（根据实际需求）
throttle = np.clip(action_raw[0], 0.0, 1.0)      # 限制在 [0, 1]
steering = np.clip(action_raw[1], -1.0, 1.0)     # 限制在 [-1, 1]

# 可选：应用增益和偏移
throttle = throttle_gain * (throttle_boost + throttle_scale * throttle)
steering = steering_gain * steering + steering_offset
```

### 5. ONNX Runtime 提供者

在 Jetson Nano 上，优先使用 GPU 加速：

```python
# 优先使用 CUDA，回退到 CPU
sess = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

**验证 GPU 可用性：**
```python
import onnxruntime as ort
print(f"Available providers: {ort.get_available_providers()}")
# 应该看到：['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 6. 常见问题

#### 问题 1：输出不匹配
- **原因**：预处理不一致、模型结构不同
- **解决**：检查预处理流程，确保与训练时完全一致

#### 问题 2：ONNX Runtime 加载失败
- **原因**：opset 版本不兼容、缺少依赖
- **解决**：使用 `opset_version=11`，检查 ONNX Runtime 安装

#### 问题 3：推理速度慢
- **原因**：使用 CPU 而非 GPU
- **解决**：确保 `CUDAExecutionProvider` 可用，检查 GPU 驱动

#### 问题 4：内存不足
- **原因**：batch size 过大、模型过大
- **解决**：使用 batch size=1，考虑模型量化

---

## 部署检查清单

在部署到 Jetson Nano 之前，确认：

- [ ] ONNX 模型验证通过（输出与 PyTorch 一致）
- [ ] 观察空间维度匹配（channels, height, width）
- [ ] 预处理流程与训练时一致
- [ ] 观察模式（obs_mode）匹配
- [ ] ONNX Runtime GPU 提供者可用
- [ ] 输入数据类型为 `float32`，范围 `[0.0, 1.0]`
- [ ] 输出动作范围正确（throttle: [0, 1], steering: [-1, 1]）

---

## 相关文件

- **转换脚本**：`export_sb3_ppo_to_onnx.py`
- **推理脚本**：`run_policy_onnx.py`
- **安装指南**：`install_onnx_nano.md`
- **验证脚本**：`experiments/check_onnx.py`

---

## 参考

- [ONNX 官方文档](https://onnx.ai/)
- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)

