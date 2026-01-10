"""Stable-Baselines3 feature extractor using a pretrained autoencoder.

This module provides a features extractor for SB3 that uses a pretrained
autoencoder encoder to compress image observations into latent vectors.
This allows using MlpPolicy instead of CnnPolicy when working with
pretrained representations.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from donkey_rl.autoencoder import load_encoder_from_checkpoint


class AELatentExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor that encodes images into latent vectors.

    This extractor wraps a pretrained autoencoder encoder to compress
    image observations (CHW float32 in [0,1]) into lower-dimensional
    latent vectors. This enables using MlpPolicy instead of CnnPolicy
    when working with pretrained visual representations.

    Example:
        To use with PPO:
        ```python
        from donkey_rl.sb3_latent_extractor import latent_policy_kwargs

        policy_kwargs = latent_policy_kwargs(
            ae_checkpoint="path/to/ae_model.pth",
            freeze=True
        )
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
        ```

    Args:
        observation_space: Gym observation space (expects image space).
        ae_checkpoint: Path to the autoencoder checkpoint file.
        freeze: If True, freeze encoder weights during training.
    """

    def __init__(
        self,
        observation_space,
        *,
        ae_checkpoint: str,
        freeze: bool = True,
    ):
        # Load encoder and config from checkpoint
        encoder, cfg = load_encoder_from_checkpoint(ae_checkpoint)

        # Initialize base class with latent dimension from config
        super().__init__(observation_space, features_dim=int(cfg.latent_dim))

        self.encoder: nn.Module = encoder

        # Freeze encoder parameters if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder.train()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode image observations into latent vectors.

        Args:
            observations: Batch of image observations with shape (B, C, H, W)
                         where values are in [0, 1] range (float32).

        Returns:
            Latent vectors with shape (B, latent_dim).
        """
        z = self.encoder(observations)
        return z


def latent_policy_kwargs(*, ae_checkpoint: str, freeze: bool = True) -> Dict[str, Any]:
    """Create SB3 policy_kwargs for training with latent features.

    This helper function constructs the policy_kwargs dictionary needed
    to use AELatentExtractor with SB3 algorithms like PPO, SAC, etc.

    Args:
        ae_checkpoint: Path to the autoencoder checkpoint file.
        freeze: If True, freeze encoder weights during training.

    Returns:
        Dictionary of policy_kwargs to pass to SB3 algorithm constructor.

    Example:
        ```python
        policy_kwargs = latent_policy_kwargs(
            ae_checkpoint="models/ae.pth",
            freeze=True
        )
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
        ```
    """
    return {
        "features_extractor_class": AELatentExtractor,
        "features_extractor_kwargs": {
            "ae_checkpoint": ae_checkpoint,
            "freeze": freeze,
        },
        "normalize_images": False,  # Images should already be normalized
    }
