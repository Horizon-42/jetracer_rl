from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import cv2


# Real-camera calibration (from real_data_process/undistort.py)
_JETRACER_MTX = np.array(
    [
        [402.60350228, 0.0, 263.30000918],
        [0.0, 537.76023089, 278.24728515],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_JETRACER_DIST_COEFFS = np.array([[-0.31085325, -0.11558236, 0.00249467, -0.00088277, 0.51442531]], dtype=np.float32)


# Real-image color correction parameters (from real_data_process/color_undistort.py)
# That file corrects with gain = 1/(1 + a*r^2 + b*r^4). Here we want the *inverse*
# effect to simulate the real camera artifact on simulator images.
_COLOR_PARAMS: Dict[str, float] = {
    "a": 0.88,
    "b": 0.88,
    "c": 0.0,
    "d": 0.0,
    "e": 0.0,
    "f": 0.0,
}


def _build_radius_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute normalized r^2 and r^4 maps (0..1)."""

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    ys, xs = np.indices((h, w), dtype=np.float32)
    x = xs - cx
    y = ys - cy
    r2 = x * x + y * y
    max_r = np.sqrt(((w - 1) / 2.0) ** 2 + ((h - 1) / 2.0) ** 2)
    max_r2 = max_r * max_r
    r2_norm = r2 / max_r2
    r4_norm = r2_norm * r2_norm
    return r2_norm, r4_norm


def _apply_inverse_color_correction_as_distortion_bgr(
    img_bgr_u8: np.ndarray,
    *,
    r2: np.ndarray,
    r4: np.ndarray,
) -> np.ndarray:
    """Apply the inverse of real-image color correction.

    real_data_process/color_undistort.py does:
      channel *= 1/(1 + p*r^2 + q*r^4)

    To simulate the artifact on simulator images, we do the reverse:
      channel *= (1 + p*r^2 + q*r^4)

    Image is assumed BGR.
    """

    a = float(_COLOR_PARAMS["a"])
    b = float(_COLOR_PARAMS["b"])
    c = float(_COLOR_PARAMS["c"])
    d = float(_COLOR_PARAMS["d"])
    e = float(_COLOR_PARAMS["e"])
    f = float(_COLOR_PARAMS["f"])

    factor_r = 1.0 + a * r2 + b * r4
    factor_g = 1.0 + c * r2 + d * r4
    factor_b = 1.0 + e * r2 + f * r4

    # Avoid extreme amplification.
    factor_r = np.clip(factor_r, 0.2, 5.0)
    factor_g = np.clip(factor_g, 0.2, 5.0)
    factor_b = np.clip(factor_b, 0.2, 5.0)

    img_f = img_bgr_u8.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = cv2.split(img_f)
    r_out = np.clip(r_ch * factor_r, 0.0, 1.0)
    g_out = np.clip(g_ch * factor_g, 0.0, 1.0)
    b_out = np.clip(b_ch * factor_b, 0.0, 1.0)
    out = cv2.merge([b_out, g_out, r_out])
    return (out * 255.0).astype(np.uint8)


class ObsPreprocess(gym.ObservationWrapper):
    """Observation preprocessing for sim->JetRacer domain gap.

    Pipeline (in order):
    1) Cache raw sim RGB frame (HWC uint8)
    2) Optional lens distortion (inverse of real-image undistort)
    3) Optional color distortion (inverse of real-image color correction)
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
        enable_color_distortion: bool = False,
    ):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        self._enable_distortion = bool(enable_distortion)
        self._enable_color_distortion = bool(enable_color_distortion)

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

        self._distort_map_key: Optional[Tuple[int, int]] = None
        self._distort_map_x: Optional[np.ndarray] = None
        self._distort_map_y: Optional[np.ndarray] = None

        self._rmap_key: Optional[Tuple[int, int]] = None
        self._r2: Optional[np.ndarray] = None
        self._r4: Optional[np.ndarray] = None

    def _get_lens_distortion_maps(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build maps so that output is *distorted* and input is *undistorted* (sim)."""

        key = (int(h), int(w))
        if self._distort_map_key == key and self._distort_map_x is not None and self._distort_map_y is not None:
            return self._distort_map_x, self._distort_map_y

        # We compute a "new" camera matrix similar to real_data_process/undistort.py.
        # In OpenCV calibration, distortion is defined w.r.t. the *original* camera matrix.
        # For "distort sim image" we want the inverse mapping of undistort:
        # for each output (distorted) pixel, find the corresponding input (undistorted) pixel.
        new_cam, _roi = cv2.getOptimalNewCameraMatrix(_JETRACER_MTX, _JETRACER_DIST_COEFFS, (w, h), 1, (w, h))

        ys, xs = np.indices((h, w), dtype=np.float32)
        pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32).reshape(-1, 1, 2)

        # Given *distorted* pixel coords (pts), compute corresponding *undistorted* pixel coords.
        # Use the original camera matrix with dist coeffs, and project into pixel coords with P=new_cam.
        # Those undistorted coords are where we should sample from the sim image.
        undist = cv2.undistortPoints(pts, _JETRACER_MTX, _JETRACER_DIST_COEFFS, P=new_cam)
        undist = undist.reshape(h, w, 2)

        map_x = undist[:, :, 0].astype(np.float32)
        map_y = undist[:, :, 1].astype(np.float32)

        self._distort_map_key = key
        self._distort_map_x = map_x
        self._distort_map_y = map_y
        return map_x, map_y

    def _get_radius_maps(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(h), int(w))
        if self._rmap_key == key and self._r2 is not None and self._r4 is not None:
            return self._r2, self._r4
        self._r2, self._r4 = _build_radius_maps(h, w)
        self._rmap_key = key
        return self._r2, self._r4

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
        if self._enable_distortion:
            try:
                h, w, _c = img.shape
                map_x, map_y = self._get_lens_distortion_maps(h, w)
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

        # 3) Color distortion (inverse of real-image correction)
        if self._enable_color_distortion:
            try:
                h, w, _c = img.shape
                r2, r4 = self._get_radius_maps(h, w)
                # apply on BGR, then convert back to RGB
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bgr = _apply_inverse_color_correction_as_distortion_bgr(bgr, r2=r2, r4=r4)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
