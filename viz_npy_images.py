from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np


def _to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    """Accept HWC uint8 RGB or CHW float and return HWC RGB uint8."""

    x = np.asarray(arr)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        if x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)
        x = np.transpose(x, (1, 2, 0))

    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Unsupported array shape: {x.shape}")

    if x.dtype == np.uint8:
        return x

    x = x.astype(np.float32)
    if float(x.max()) <= 1.5:
        x = x * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump .npy frames in a folder to PNG previews.")
    p.add_argument("dir", type=str, help="Folder containing .npy frames")
    p.add_argument("--out", type=str, default="viz_out", help="Output folder for PNG previews")
    p.add_argument("--max", type=int, default=16, help="How many images to export")
    args = p.parse_args()

    in_dir = str(args.dir)
    out_dir = str(args.out)
    os.makedirs(out_dir, exist_ok=True)

    files: List[str] = sorted(
        os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.lower().endswith(".npy")
    )
    if not files:
        raise SystemExit(f"No .npy files found under: {in_dir}")

    import cv2

    n = min(int(args.max), len(files))
    for i, path in enumerate(files[:n]):
        rgb = _to_rgb_u8(np.load(path))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, f"viz_{i:03d}.png")
        cv2.imwrite(out_path, bgr)

    print(f"Saved {n} previews to: {out_dir}")


if __name__ == "__main__":
    main()
