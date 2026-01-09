import torch
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

MODEL_ZIP = "models/JetRacer_20/best_model.zip"
OUTPUT_ONNX = "ppo_policy.onnx"
OPSET = 11


def make_dummy_obs(obs_space):
    """
    根据 observation_space 自动构造 dummy obs
    """
    if isinstance(obs_space, spaces.Box):
        shape = obs_space.shape
        dummy = torch.randn((1,) + shape)
        return dummy

    elif isinstance(obs_space, spaces.Dict):
        dummy = {}
        for k, space in obs_space.spaces.items():
            dummy[k] = make_dummy_obs(space)
        return dummy

    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(obs_space)}"
        )


def main():
    model = PPO.load(MODEL_ZIP, device="cpu")
    policy = model.policy
    policy.eval()

    obs_space = model.observation_space
    print("Observation space:", obs_space)
    print("Policy type:", type(policy))

    dummy_obs = make_dummy_obs(obs_space)

    # forward dry-run（强烈建议保留）
    with torch.no_grad():
        out = policy(dummy_obs)
        print("Policy forward output:", out)

    # ONNX 输入名
    if isinstance(obs_space, spaces.Dict):
        input_names = list(dummy_obs.keys())
        dynamic_axes = {
            k: {0: "batch"} for k in input_names
        }
    else:
        input_names = ["obs"]
        dynamic_axes = {"obs": {0: "batch"}}

    torch.onnx.export(
        policy,
        dummy_obs,
        OUTPUT_ONNX,
        opset_version=OPSET,
        input_names=input_names,
        output_names=["actions", "values", "log_probs"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ Exported ONNX model to {OUTPUT_ONNX}")


if __name__ == "__main__":
    main()
