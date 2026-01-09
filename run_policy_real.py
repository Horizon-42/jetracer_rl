import argparse
import os
import sys
import time
from typing import Iterator, List, Optional, Sequence, Tuple

# Python 3.6/3.7 pickle backport to support protocol 5 (used by newer torch/SB3 models)
if sys.version_info < (3, 8):
    try:
        import pickle5
        import pickle
        # Monkeypatch pickle to use pickle5's loader, but keep original module
        # This allows 'import pickle' to still provide private attrs like _getattribute (needed by torch)
        pickle.Unpickler = pickle5.Unpickler
        pickle.load = pickle5.load
        pickle.loads = pickle5.loads
    except ImportError:
        pass

# Shim for gymnasium (if missing, alias it to gym)
# Newer SB3 models pickle references to 'gymnasium', but on Jetson (Py3.6) we might only have 'gym'.
try:
    import gymnasium
except ImportError:
    try:
        import gym
        sys.modules["gymnasium"] = gym
    except ImportError:
        pass

import numpy as np


class Action(object):
    __slots__ = ("throttle", "steering")

    def __init__(self, throttle: float, steering: float):
        self.throttle = float(throttle)
        self.steering = float(steering)


def _clip_action(a: Action) -> Action:
    thr = float(np.clip(a.throttle, -0.5, 1.0))
    steer = float(np.clip(a.steering, -1.0, 1.0))
    return Action(throttle=thr, steering=steer)


def _load_sb3_model(path: str, *, obs_width: int, obs_height: int):
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}")

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as e:
        raise SystemExit("stable-baselines3 not installed in this environment") from e

    # Jetson/Nano environments sometimes end up with a broken/unpickled observation_space
    # (often due to gym/gymnasium/numpy version mismatches). SB3 then crashes while
    # computing CNN shapes during load.
    #
    # Workaround: explicitly provide the expected spaces.
    # Need to verify if 'gym' is available from the shim or import
    try:
        import gym
    except ImportError:
        pass

    # Explicitly reconstruct spaces to avoid pickle deserialization issues (deque error, etc).
    # This also ensures we use the correct types (uint8 for images) which NatureCNN expects.
    # We use standard gym.spaces.Box which is compatible with valid gym versions.
    
    # 1. Observation Space: (3, H, W) in [0, 255] uint8
    obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(3, int(obs_height), int(obs_width)),
        dtype=np.uint8,
    )

    # 2. Action Space: (2,) in [-1, 1] float32 (Steering, Throttle)
    # Note: We assume standard continuous control PPO
    act_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    )

    def _do_load(*, extra_custom_objects: Optional[dict] = None):
        # Override the spaces saved in the model file with our manually constructed ones.
        # This fixes the "AttributeError: 'collections.deque' object has no attribute 'seed'"
        # and ensuring NatureCNN gets a uint8 space.
        custom_objects = {
            "observation_space": obs_space,
            "action_space": act_space,
        }
        if extra_custom_objects:
            custom_objects.update(extra_custom_objects)
        
        # We pass env=None because correct spaces are now injected via custom_objects
        return PPO.load(path, device="auto", env=None, custom_objects=custom_objects)

    try:
        return _do_load()
    except RuntimeError as e:
        msg = str(e)
        if "Could not infer dtype of numpy.float32" not in msg:
            raise

        # Fallback: SB3's default NatureCNN uses observation_space.sample() to call CNN.
        # If standard Box.sample() is broken on this system, patch the extractor.
        try:
            import torch as th
            import torch.nn as nn
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            class SafeNatureCNN(BaseFeaturesExtractor):
                def __init__(self, observation_space, features_dim: int = 512):
                    super().__init__(observation_space, features_dim)
                    n_input_channels = int(observation_space.shape[0])
                    self.cnn = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    with th.no_grad():
                        sample = th.zeros((1,) + tuple(observation_space.shape), dtype=th.float32)
                        n_flatten = int(self.cnn(sample).shape[1])
                    self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

                def forward(self, observations: th.Tensor) -> th.Tensor:
                    return self.linear(self.cnn(observations))

            # Preserve the most common policy kwargs in this repo.
            safe_policy_kwargs = {
                "normalize_images": False,
                "features_extractor_class": SafeNatureCNN,
            }

            print("NOTE: Falling back to SafeNatureCNN to avoid gym Box.sample() dtype crash during model load.")
            return _do_load(extra_custom_objects={"policy_kwargs": safe_policy_kwargs})
        except Exception:
            raise


def _predict_action(model, obs_chw_float01: np.ndarray, *, deterministic: bool) -> Action:
    act, _state = model.predict(obs_chw_float01, deterministic=deterministic)
    act = np.asarray(act, dtype=np.float32).reshape(-1)
    if act.shape[0] != 2:
        raise RuntimeError(f"Expected 2-dim action [throttle, steering], got shape {act.shape}")
    return _clip_action(Action(throttle=float(act[0]), steering=float(act[1])))


def _iter_image_files(images_dir: str) -> Sequence[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []  # type: List[str]
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if not os.path.isfile(p):
            continue
        _, ext = os.path.splitext(name.lower())
        if ext in exts:
            files.append(p)
    return files


def _open_camera(*, camera_index: int, camera_gst: str = ""):
    import cv2

    if camera_gst.strip():
        cap = cv2.VideoCapture(camera_gst.strip(), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera: index={camera_index} gst={'yes' if camera_gst else 'no'}")
    return cap


def _iter_frames_from_camera(*, camera_index: int, camera_gst: str = "") -> Iterator[np.ndarray]:
    import cv2

    cap = _open_camera(camera_index=camera_index, camera_gst=camera_gst)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            yield frame_bgr
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _iter_frames_from_dir(images_dir: str, *, loop: bool) -> Iterator[Tuple[str, np.ndarray]]:
    import cv2

    files = _iter_image_files(images_dir)
    if not files:
        raise SystemExit(f"No images found in: {images_dir}")

    while True:
        for p in files:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield p, img
        if not loop:
            break


class _JetRacerActuator:
    def __init__(self):
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Failed to import jetracer control library. On Jetson Nano, install `jetracer` (NvidiaRacecar)."
            ) from e

        self._car = NvidiaRacecar()

    def apply(self, action: Action) -> None:
        action = _clip_action(action)
        self._car.steering = float(action.steering)
        self._car.throttle = float(action.throttle)

    def stop(self) -> None:
        try:
            self._car.throttle = 0.0
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an SB3 policy on the real JetRacer (no simulator code).")
    )

    p.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["real", "mock"],
        help=(
            "real: read camera and actuate the car\n"
            "mock: read camera OR an image folder, and only print actions"
        ),
    )

    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")

    # Input source
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture device index")
    p.add_argument(
        "--camera-gst",
        type=str,
        default="",
        help="Optional GStreamer pipeline string for Jetson CSI cameras (if set, overrides --camera index).",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default="",
        help="Mock mode: directory of images (jpg/png). If set, reads images instead of camera.",
    )
    p.add_argument("--loop", action="store_true", help="Mock mode + --images-dir: loop forever")

    # Runtime
    p.add_argument("--fps", type=float, default=15.0, help="Control loop FPS")
    p.add_argument("--show", action="store_true", help="Show preview window (requires display)")

    # Preprocess (must match training)
    p.add_argument("--obs-width", type=int, default=84)
    p.add_argument("--obs-height", type=int, default=84)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Keep imports here so the script starts even if OpenCV is missing, and fails with a clear error later.
    import cv2

    from donkey_rl.real_obs_preprocess import preprocess_real_frame_bgr_to_chw_float01

    model = _load_sb3_model(str(args.model), obs_width=int(args.obs_width), obs_height=int(args.obs_height))

    actuator: Optional[_JetRacerActuator] = None
    if str(args.mode) == "real":
        actuator = _JetRacerActuator()

    dt = 1.0 / max(1e-6, float(args.fps))

    def handle_action(action: Action, *, frame_tag: str = "") -> None:
        if str(args.mode) == "mock":
            prefix = f"{frame_tag} " if frame_tag else ""
            print(f"{prefix}action throttle={action.throttle:.3f} steer={action.steering:.3f}")
            return
        assert actuator is not None
        actuator.apply(action)

    try:
        if str(args.mode) == "mock" and str(args.images_dir).strip():
            for path, frame_bgr in _iter_frames_from_dir(str(args.images_dir), loop=bool(args.loop)):
                t0 = time.time()

                obs = preprocess_real_frame_bgr_to_chw_float01(
                    frame_bgr,
                    out_width=int(args.obs_width),
                    out_height=int(args.obs_height),
                )
                action = _predict_action(model, obs, deterministic=bool(args.deterministic))
                handle_action(action, frame_tag=os.path.basename(path))

                if bool(args.show):
                    cv2.imshow("input", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        else:
            # real mode: always camera
            # mock mode: camera if --images-dir is not provided
            for frame_bgr in _iter_frames_from_camera(camera_index=int(args.camera), camera_gst=str(args.camera_gst)):
                t0 = time.time()

                obs = preprocess_real_frame_bgr_to_chw_float01(
                    frame_bgr,
                    out_width=int(args.obs_width),
                    out_height=int(args.obs_height),
                )
                action = _predict_action(model, obs, deterministic=bool(args.deterministic))
                handle_action(action)

                if bool(args.show):
                    cv2.imshow("camera", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        if actuator is not None:
            actuator.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
