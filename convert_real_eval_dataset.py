from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import numpy as np

from donkey_rl.real_obs_preprocess import preprocess_real_frame_bgr_to_chw_float01


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert real JetRacer road images (e.g. 640x480 JPG) into the same format "
            "used for training/eval: 84x84 RGB uint8 frames saved as .npy."
        )
    )

    p.add_argument("--in-dir", type=str, default="data/road")
    p.add_argument("--out-dir", type=str, default="datasets/eval_real")

    p.add_argument("--out-width", type=int, default=84)
    p.add_argument("--out-height", type=int, default=84)

    p.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Glob pattern inside in-dir (e.g. '*.jpg' or '*.png').",
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save one converted frame every N input images (useful for subsampling).",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, stop after converting this many images.",
    )

    return p.parse_args()


def _list_files(in_dir: str, pattern: str) -> List[str]:
    import glob

    files = sorted(glob.glob(os.path.join(in_dir, pattern)))
    return files


def main() -> None:
    args = _parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = _list_files(args.in_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files matched {args.pattern} under {args.in_dir}")

    meta = {
        "time": time.time(),
        "args": vars(args),
        "n_input_files": len(files),
        "format": "frame_XXXXXXXX.npy is HWC RGB uint8 (after undistort+color correction+resize)",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    import cv2

    saved = 0
    for i, path in enumerate(files):
        if args.limit and saved >= int(args.limit):
            break
        if int(args.save_every) > 1 and (i % int(args.save_every) != 0):
            continue

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        chw = preprocess_real_frame_bgr_to_chw_float01(
            img_bgr,
            out_width=int(args.out_width),
            out_height=int(args.out_height),
        )

        # Save as HWC RGB uint8 (same on-disk format as collect_obs_dataset.py)
        rgb_u8 = (np.clip(chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_path = os.path.join(args.out_dir, f"frame_{saved:08d}.npy")
        np.save(out_path, rgb_u8)
        saved += 1

        if saved % 1000 == 0:
            print(f"converted={saved} last={path}")

    print(f"Done. Converted {saved} frames into: {args.out_dir}")


if __name__ == "__main__":
    main()
