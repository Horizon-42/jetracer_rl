import argparse
import os
import sys
import numpy as np
import torch as th
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO

# --- 1. 定义导出包装器 ---
class OnnxablePolicy(th.nn.Module):
    """
    包装 SB3 的策略，只输出确定性的动作 (Deterministic Action / Mean)。
    去掉了 Value Net 和 Log Prob 计算，只保留 Actor 部分。
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # deterministic=True 让 SB3 返回高斯分布的均值 (Mean)
        # policy() 返回 (actions, values, log_probs)，我们只需要 actions [0]
        return self.policy(observation, deterministic=True)[0]

def export_model(model_path, output_path, obs_shape):
    print(f"Loading SB3 model from: {model_path}")
    
    # 强制加载到 CPU，避免设备不匹配问题
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 创建包装后的模型
    onnx_policy = OnnxablePolicy(model.policy)
    
    # 创建 Dummy Input (根据你的观察空间，通常是 1, 3, H, W)
    # 注意：这里假设输入是 float32 (已经归一化)。
    # 如果你的模型包含预处理层（比如 Normalize），输入可能是 uint8。
    # 大多数 SB3 CNN 策略内部都接受 float 输入。
    dummy_input = th.randn(1, *obs_shape)

    print(f"Exporting to ONNX: {output_path} ...")
    th.onnx.export(
        onnx_policy,
        dummy_input,
        output_path,
        opset_version=11,          # Jetson/Py3.6 建议使用 opset 11
        input_names=["input"],     # 输入节点名称
        output_names=["action"],   # 输出节点名称
        dynamic_axes={
            "input": {0: "batch_size"},  # 支持动态 Batch Size
            "action": {0: "batch_size"}
        }
    )
    print("Export complete.")
    
    # --- 验证环节 ---
    verify_export(onnx_policy, output_path, dummy_input)

def verify_export(torch_model, onnx_path, dummy_input):
    print("\nVerifying exported model...")
    
    # 1. PyTorch 推理
    with th.no_grad():
        torch_out = torch_model(dummy_input).numpy()
    
    # 2. ONNX Runtime 推理
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

    # 3. 对比
    # 允许一定的浮点误差 (atol=1e-5)
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
        print("✅ SUCCESS: ONNX model matches PyTorch model outputs.")
        print(f"   Max absolute difference: {np.max(np.abs(torch_out - onnx_out)):.8f}")
    except AssertionError as e:
        print("❌ WARNING: Model outputs mismatch!")
        print(e)

def main():
    parser = argparse.ArgumentParser(description="Convert SB3 PPO model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--output", type=str, default=".", help="Output folder path (default: current directory)")
    parser.add_argument("--height", type=int, default=84, help="Observation height (default: 84)")
    parser.add_argument("--width", type=int, default=84, help="Observation width (default: 84)")
    parser.add_argument("--channels", type=int, default=3, help="Observation channels (default: 3)")
    
    args = parser.parse_args()
    
    # 从模型路径中提取最后一层文件夹名
    # 例如: models/scratch_centerline_v4_exp01/last_model.zip -> scratch_centerline_v4_exp01
    model_dir = os.path.dirname(os.path.abspath(args.model))
    folder_name = os.path.basename(model_dir)
    
    # 构建输出文件路径
    os.makedirs(args.output, exist_ok=True)
    output_filename = f"{folder_name}.onnx"
    output_path = os.path.join(args.output, output_filename)
    
    print(f"Input model: {args.model}")
    print(f"Output folder: {args.output}")
    print(f"Output filename: {output_filename}")
    print(f"Full output path: {output_path}\n")
    
    obs_shape = (args.channels, args.height, args.width)
    export_model(args.model, output_path, obs_shape)

if __name__ == "__main__":
    main()