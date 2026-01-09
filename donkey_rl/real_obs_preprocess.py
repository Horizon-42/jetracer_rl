# Python 3.6 compatible (no __future__ annotations)

from typing import Dict, Tuple

import cv2
import numpy as np


# Real-camera calibration (from real_data_process/undistort.py)
# Calibrated on 640x480.
_CALIB_W = 640
_CALIB_H = 480

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
# These are the *correction* gains: gain = 1/(1 + p*r^2 + q*r^4)
_COLOR_PARAMS: Dict[str, float] = {
    "a": 0.88,
    "b": 0.88,
    "c": 0.0,
    "d": 0.0,
    "e": 0.0,
    "f": 0.0,
}


def _scaled_camera_matrix(*, w: int, h: int) -> np.ndarray:
    sx = float(w) / float(_CALIB_W)
    sy = float(h) / float(_CALIB_H)
    m = _JETRACER_MTX.copy()
    m[0, 0] *= sx  # fx
    m[1, 1] *= sy  # fy
    m[0, 2] *= sx  # cx
    m[1, 2] *= sy  # cy
    return m


def undistort_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Geometric undistortion for JetRacer camera frames.

    Input/output are BGR uint8.
    """

    h, w = img_bgr.shape[:2]
    mtx = _scaled_camera_matrix(w=w, h=h)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, _JETRACER_DIST_COEFFS, (w, h), 1, (w, h))

    undistorted = cv2.undistort(img_bgr, mtx, _JETRACER_DIST_COEFFS, None, newcameramtx)

    # Match your original script: crop to ROI.
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]

    return undistorted


def build_radius_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute r^2 and r^4 over the image with r normalized to [0,1]."""

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


def apply_color_lens_correction_bgr(img_bgr: np.ndarray, *, r2: np.ndarray, r4: np.ndarray) -> np.ndarray:
    """Apply your real-image per-channel radial correction (BGR).

    gain_R = 1 / (1 + a*r^2 + b*r^4)
    gain_G = 1 / (1 + c*r^2 + d*r^4)
    gain_B = 1 / (1 + e*r^2 + f*r^4)
    """

    a = float(_COLOR_PARAMS["a"])
    b = float(_COLOR_PARAMS["b"])
    c = float(_COLOR_PARAMS["c"])
    d = float(_COLOR_PARAMS["d"])
    e = float(_COLOR_PARAMS["e"])
    f = float(_COLOR_PARAMS["f"])

    gain_r = 1.0 / (1.0 + a * r2 + b * r4)
    gain_g = 1.0 / (1.0 + c * r2 + d * r4)
    gain_b = 1.0 / (1.0 + e * r2 + f * r4)

    gain_r = np.clip(gain_r, 0.2, 5.0)
    gain_g = np.clip(gain_g, 0.2, 5.0)
    gain_b = np.clip(gain_b, 0.2, 5.0)

    img_f = img_bgr.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = cv2.split(img_f)

    r_corr = np.clip(r_ch * gain_r, 0.0, 1.0)
    g_corr = np.clip(g_ch * gain_g, 0.0, 1.0)
    b_corr = np.clip(b_ch * gain_b, 0.0, 1.0)

    corrected = cv2.merge([b_corr, g_corr, r_corr])
    return (corrected * 255.0).astype(np.uint8)


def preprocess_real_frame_bgr_to_chw_float01(
    img_bgr_u8: np.ndarray,
    *,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Full A-route real-car preprocessing.

    1) undistort (geometric)
    2) color lens correction
    3) resize
    4) convert to RGB CHW float32 in [0,1]

    Returns: np.ndarray shape (3, out_height, out_width)
    """

    img_bgr_u8 = np.asarray(img_bgr_u8)
    if img_bgr_u8.dtype != np.uint8:
        img_bgr_u8 = np.clip(img_bgr_u8, 0, 255).astype(np.uint8)

    undist = undistort_bgr(img_bgr_u8)

    h, w = undist.shape[:2]
    r2, r4 = build_radius_maps(h, w)
    corrected = apply_color_lens_correction_bgr(undist, r2=r2, r4=r4)

    resized = cv2.resize(corrected, (int(out_width), int(out_height)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw
