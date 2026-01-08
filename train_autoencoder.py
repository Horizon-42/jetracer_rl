from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from donkey_rl.autoencoder import AEConfig, ConvAutoEncoder, save_ae_checkpoint


class NpyFrameDataset(Dataset):
    def __init__(self, files: List[str]):
        self._files = list(files)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self._files[idx])
        # Expect HWC RGB uint8
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        x = arr.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        return torch.from_numpy(x)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a convolutional autoencoder on collected sim frames.")
    p.add_argument("--data-dir", type=str, default="datasets/obs_ae")
    p.add_argument("--out-dir", type=str, default="ae_runs")

    p.add_argument("--img-w", type=int, default=84)
    p.add_argument("--img-h", type=int, default=84)
    p.add_argument("--latent-dim", type=int, default=64)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--save-every-epochs", type=int, default=1)
    return p.parse_args()


def _split(files: List[str], val_split: float, seed: int) -> Tuple[List[str], List[str]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(max(1, round(len(files) * float(val_split))))
    return files[n_val:], files[:n_val]


def main() -> None:
    args = _parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    files = sorted(glob.glob(os.path.join(args.data_dir, "frame_*.npy")))
    if not files:
        raise SystemExit(f"No .npy frames found under {args.data_dir}. Run collect_obs_dataset.py first.")

    train_files, val_files = _split(files, val_split=float(args.val_split), seed=int(args.seed))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"ae_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_files": len(files)}, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cfg = AEConfig(in_channels=3, img_h=int(args.img_h), img_w=int(args.img_w), latent_dim=int(args.latent_dim))
    model = ConvAutoEncoder(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        NpyFrameDataset(train_files),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        NpyFrameDataset(val_files),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        n = 0
        for x in train_loader:
            x = x.to(device)
            _z, recon = model(x)
            loss = loss_fn(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1

        train_loss = total / max(1, n)

        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                _z, recon = model(x)
                loss = loss_fn(recon, x)
                vtotal += float(loss.item())
                vn += 1
        val_loss = vtotal / max(1, vn)

        print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_ae_checkpoint(os.path.join(out_dir, "best_ae.pt"), model=model)

        if int(args.save_every_epochs) > 0 and (epoch % int(args.save_every_epochs) == 0):
            save_ae_checkpoint(os.path.join(out_dir, f"ae_epoch_{epoch:03d}.pt"), model=model)

    save_ae_checkpoint(os.path.join(out_dir, "last_ae.pt"), model=model)
    print(f"Saved AE checkpoints to: {out_dir}")


if __name__ == "__main__":
    main()
