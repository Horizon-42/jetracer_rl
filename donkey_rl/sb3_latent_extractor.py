from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from donkey_rl.autoencoder import load_encoder_from_checkpoint


class AELatentExtractor(BaseFeaturesExtractor):
    """SB3 features extractor that encodes images into a latent vector via a pretrained AE encoder.

    The environment should still emit image observations (CHW float32 in [0,1]).
    This extractor turns them into a latent vector so you can use MlpPolicy.
    """

    def __init__(
        self,
        observation_space,
        *,
        ae_checkpoint: str,
        freeze: bool = True,
    ):
        # Determine latent dim from checkpoint.
        encoder, cfg = load_encoder_from_checkpoint(ae_checkpoint)
        super().__init__(observation_space, features_dim=int(cfg.latent_dim))

        self.encoder: nn.Module = encoder
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.encoder.eval() if freeze else self.encoder.train()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B,C,H,W), float in [0,1]
        z = self.encoder(observations)
        return z


def latent_policy_kwargs(*, ae_checkpoint: str, freeze: bool = True) -> Dict[str, Any]:
    """Helper to build SB3 policy_kwargs for latent training."""

    return {
        "features_extractor_class": AELatentExtractor,
        "features_extractor_kwargs": {"ae_checkpoint": ae_checkpoint, "freeze": freeze},
        "normalize_images": False,
    }
