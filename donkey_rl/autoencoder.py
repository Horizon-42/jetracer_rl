from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AEConfig:
    in_channels: int = 3
    img_h: int = 84
    img_w: int = 84
    latent_dim: int = 64


class ConvEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.img_h, cfg.img_w)
            feat = self.net(dummy)
            self._feat_shape = tuple(feat.shape[1:])
            feat_dim = int(feat.numel())

        self.fc = nn.Linear(feat_dim, cfg.latent_dim)

    @property
    def feat_shape(self) -> Tuple[int, int, int]:
        return self._feat_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = z.flatten(start_dim=1)
        z = self.fc(z)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, cfg: AEConfig, feat_shape: Tuple[int, int, int]):
        super().__init__()
        c, h, w = feat_shape
        self.cfg = cfg
        self._feat_shape = feat_shape
        self.fc = nn.Linear(cfg.latent_dim, c * h * w)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, cfg.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        c, h, w = self._feat_shape
        x = x.view(z.shape[0], c, h, w)
        x = self.net(x)
        if x.shape[-2:] != (self.cfg.img_h, self.cfg.img_w):
            x = F.interpolate(x, size=(self.cfg.img_h, self.cfg.img_w), mode="bilinear", align_corners=False)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg)
        self.decoder = ConvDecoder(cfg, feat_shape=self.encoder.feat_shape)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def save_ae_checkpoint(path: str, *, model: ConvAutoEncoder) -> None:
    payload: Dict = {
        "cfg": asdict(model.cfg),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_encoder_from_checkpoint(path: str) -> Tuple[ConvEncoder, AEConfig]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = AEConfig(**ckpt["cfg"])
    model = ConvAutoEncoder(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.encoder, cfg
