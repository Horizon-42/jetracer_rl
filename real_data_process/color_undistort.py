import cv2
import numpy as np
from typing import Tuple, Dict

def build_radius_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute r^2 and r^4 over the image, where r is normalized radius from image center.
    r = sqrt((x - cx)^2 + (y - cy)^2) / max_radius,  0 <= r <= 1
    """
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


def apply_color_lens_correction(
    img: np.ndarray,
    params: Dict[str, float],
    r2: np.ndarray,
    r4: np.ndarray,
) -> np.ndarray:
    """
    Apply per-channel radial gain:
      gain_R = 1 / (1 + a*r^2 + b*r^4)
      gain_G = 1 / (1 + c*r^2 + d*r^4)
      gain_B = 1 / (1 + e*r^2 + f*r^4)
    Image is assumed BGR (OpenCV default).
    """
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    e = params["e"]
    f = params["f"]

    # Compute per-channel gains
    gain_R = 1.0 / (1.0 + a * r2 + b * r4)
    gain_G = 1.0 / (1.0 + c * r2 + d * r4)
    gain_B = 1.0 / (1.0 + e * r2 + f * r4)

    # Clip gains to avoid extreme amplification
    gain_R = np.clip(gain_R, 0.2, 5.0)
    gain_G = np.clip(gain_G, 0.2, 5.0)
    gain_B = np.clip(gain_B, 0.2, 5.0)

    img_f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_f)

    r_corr = r * gain_R
    g_corr = g * gain_G
    b_corr = b * gain_B

    corrected = cv2.merge(
        [
            np.clip(b_corr, 0.0, 1.0),
            np.clip(g_corr, 0.0, 1.0),
            np.clip(r_corr, 0.0, 1.0),
        ]
    )

    corrected_u8 = (corrected * 255.0).astype(np.uint8)
    return corrected_u8

# undistort parameters
params = {
    "a": 0.88, "b": 0.88,
    "c": 0.0, "d": 0.0,
    "e": 0.0, "f": 0.0
}