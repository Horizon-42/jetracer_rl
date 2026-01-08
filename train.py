import gymnasium as gym
import gym as old_gym
import gym_donkeycar
import torch
import shimmy # 确保 shimmy 已注册到 gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def _patch_old_gym_render_mode() -> None:
    """Compat for gym==0.21 used by gym-donkeycar.

    Gymnasium+Shimmy expects the underlying OpenAI Gym env to have a
    `render_mode` attribute. gym==0.21 envs typically don't, which causes:
    AttributeError: '<Env>' object has no attribute 'render_mode'
    """

    if getattr(old_gym.make, "__name__", "") == "_make_with_render_mode":
        return

    original_make = old_gym.make

    def _make_with_render_mode(*args, **kwargs):
        env = original_make(*args, **kwargs)
        try:
            getattr(env, "render_mode")
        except AttributeError:
            try:
                setattr(env, "render_mode", None)
            except Exception:
                pass
        return env

    old_gym.make = _make_with_render_mode

# 检查 CUDA 是否可用 (防止你以为在用显卡其实在跑 CPU)
if torch.cuda.is_available():
    print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not found. Training will be slow on CPU.")

if __name__ == "__main__":
    _patch_old_gym_render_mode()
    
    # --- 1. 环境配置 ---
    # 请修改为你的模拟器路径
    EXE_PATH = "remote" 
    PORT = 9091

    conf = {
        "exe_path": EXE_PATH,
        "host": "127.0.0.1",
        "port": PORT,
        "body_style": "donkey",
        "body_rgb": (255, 165, 0), # 橙色小车
        "car_name": "Shimmy_Agent",
        "font_size": 100
    }

    # --- 2. 创建环境 (Shimmy Magic) ---
    # 核心变化：
    # 1. 使用 gymnasium.make 加载 "GymV26Environment-v0"
    # 2. 将真实的 donkey 环境 ID 传给 env_id
    # 3. 将 donkey 的配置传给 make_kwargs
    # Shimmy 会自动处理 4参数(Old) -> 5参数(New) 的转换
    
    print("正在连接模拟器...")
    env = gym.make(
        "GymV21Environment-v0",
        env_id="donkey-generated-roads-v0",
        make_kwargs={"conf": conf},
    )

    # --- 3. 模型训练 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./logs/',
        name_prefix='shimmy_ppo'
    )

    # 实例化模型
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        batch_size=64,
        ent_coef=0.01, # 增加一点熵系数，鼓励探索
        tensorboard_log="./tensorboard_logs/"
    )

    print("开始训练! 也可以在终端运行 'tensorboard --logdir ./tensorboard_logs' 查看进度")
    
    try:
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
        model.save("donkey_ppo_cuda12")
        print("训练完成。")
    except KeyboardInterrupt:
        print("检测到中断，正在保存模型...")
        model.save("donkey_ppo_interrupted")
    finally:
        env.close()
        print("环境已关闭。")