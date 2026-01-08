from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import cv2


def _build_radial_distortion_map(
    *,
    height: int,
    width: int,
    k1: float,
    k2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create cv2.remap maps that apply a simple radial distortion.

    This is not a physically accurate camera model; it's a lightweight approximation
    intended to make simulator images "feel" closer to a real JetRacer camera.

    Coordinates are normalized to [-1, 1]. Distortion model:
      x_d = x * (1 + k1*r^2 + k2*r^4)
      y_d = y * (1 + k1*r^2 + k2*r^4)

    Positive k1 typically yields barrel distortion; negative yields pincushion.
    """

    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x, y = np.meshgrid(xs, ys)

    r2 = x * x + y * y
    scale = 1.0 + float(k1) * r2 + float(k2) * (r2 * r2)

    xd = x * scale
    yd = y * scale

    # Map distorted normalized coords back to pixel coords.
    map_x = ((xd + 1.0) * 0.5 * (width - 1)).astype(np.float32)
    map_y = ((yd + 1.0) * 0.5 * (height - 1)).astype(np.float32)

    return map_x, map_y


def _apply_red_edge_bias(
    img_rgb_u8: np.ndarray,
    *,
    strength: float,
    power: float,
) -> np.ndarray:
    """Add a red tint that increases towards the image edges."""

    if strength <= 0:
        return img_rgb_u8

    h, w, _c = img_rgb_u8.shape
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x, y = np.meshgrid(xs, ys)

    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0.0, 1.0)
    weight = (r**float(power)) * float(strength)

    out = img_rgb_u8.astype(np.float32)
    # RGB: boost red channel (index 0)
    out[:, :, 0] = np.clip(out[:, :, 0] + weight * 255.0, 0.0, 255.0)
    return out.astype(np.uint8)


class ObsPreprocess(gym.ObservationWrapper):
    """Observation preprocessing for sim->JetRacer domain gap.

    Pipeline (in order):
    1) Cache raw sim RGB frame (HWC uint8)
    2) Optional lens distortion (radial remap)
    3) Optional color distortion (red edge bias)
    4) Resize to (width, height)
    5) Convert to CHW float32 in [0, 1] for SB3 CnnPolicy

    Debug caches (for DebugObsDumpCallback):
    - last_raw_observation: raw sim frame (HWC uint8)
    - last_distorted_observation: after distort/color (HWC uint8)
    - last_resized_observation: after resize, before normalize (HWC uint8)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        width: int = 84,
        height: int = 84,
        enable_distortion: bool = False,
        distortion_k1: float = 0.0,
        distortion_k2: float = 0.0,
        enable_color_distortion: bool = False,
        red_edge_strength: float = 0.0,
        red_edge_power: float = 2.0,
    ):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        self._enable_distortion = bool(enable_distortion)
        self._k1 = float(distortion_k1)
        self._k2 = float(distortion_k2)

        self._enable_color_distortion = bool(enable_color_distortion)
        self._red_strength = float(red_edge_strength)
        self._red_power = float(red_edge_power)

        # Expect original observations to be HWC RGB.
        _h, _w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        # Output: CHW float32 [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

        self.last_raw_observation: Optional[np.ndarray] = None
        self.last_distorted_observation: Optional[np.ndarray] = None
        self.last_resized_observation: Optional[np.ndarray] = None

        self._map_cache_key: Optional[Tuple[int, int, float, float]] = None
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None

    def _get_maps(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(h), int(w), float(self._k1), float(self._k2))
        if self._map_cache_key != key or self._map_x is None or self._map_y is None:
            self._map_x, self._map_y = _build_radial_distortion_map(height=h, width=w, k1=self._k1, k2=self._k2)
            self._map_cache_key = key
        return self._map_x, self._map_y

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # 1) Cache raw
        try:
            raw = np.asarray(observation)
            if raw.dtype != np.uint8:
                raw = np.clip(raw, 0, 255).astype(np.uint8)
            self.last_raw_observation = raw.copy()
        except Exception:
            self.last_raw_observation = None
            raw = np.asarray(observation)

        img = raw

        # 2) Lens distortion
        if self._enable_distortion and (self._k1 != 0.0 or self._k2 != 0.0):
            try:
                h, w, _c = img.shape
                map_x, map_y = self._get_maps(h, w)
                img = cv2.remap(
                    img,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            except Exception:
                # If something goes wrong, keep original.
                img = raw

        # 3) Color distortion (red edge bias)
        if self._enable_color_distortion and self._red_strength > 0.0:
            try:
                img = _apply_red_edge_bias(img, strength=self._red_strength, power=self._red_power)
            except Exception:
                pass

        # Cache distorted
        try:
            self.last_distorted_observation = np.asarray(img).copy()
        except Exception:
            self.last_distorted_observation = None

        # 4) Resize
        resized = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        try:
            self.last_resized_observation = np.asarray(resized).copy()
        except Exception:
            self.last_resized_observation = None

        # 5) Normalize to CHW float32 [0,1]
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw
