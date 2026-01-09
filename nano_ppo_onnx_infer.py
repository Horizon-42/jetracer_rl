import onnxruntime as ort
import numpy as np
import cv2

# ====== 配置区 ======
ONNX_MODEL = "ppo_policy.onnx"
IMG_PATH = "road_000.jpg"   # 84x84 的测试图片
CHANNELS = 3            # 改成 1 如果你是灰度图
# ====================

def preprocess_image(img_path):
    """
    读取图片并转换为 SB3 CnnPolicy 期望的格式
    """
    img = cv2.imread(img_path)

    # resize
    img = cv2.resize(img, (84, 84))

    if CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]  # (H, W, 1)

    # uint8 -> float32 & 归一化
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # batch 维
    img = np.expand_dims(img, axis=0)

    return img


def main():
    # 1️⃣ 加载 ONNX
    sess = ort.InferenceSession(
        ONNX_MODEL,
        providers=["CUDAExecutionProvider"]
    )

    print("ONNX inputs:", sess.get_inputs())
    print("ONNX outputs:", sess.get_outputs())

    # 2️⃣ 构造输入
    obs = preprocess_image(IMG_PATH)

    # 3️⃣ 推理
    outputs = sess.run(
        None,
        {sess.get_inputs()[0].name: obs}
    )

    # PPO policy 默认输出
    actions, values, log_probs = outputs

    print("Raw action output:", actions)
    print("Value:", values)

    # 4️⃣ 动作后处理（非常关键）
    # 连续动作：直接用
    action = actions[0]

    print("Final action:", action)


if __name__ == "__main__":
    main()
