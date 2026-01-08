"""Helper modules for training DonkeyCar policies with SB3.

This package intentionally keeps modules small and easy to modify:
- compat: monkeypatches for gym/gymnasium/shimmy interop
- env: DonkeyCar env creation/config
- wrappers: action + observation wrappers
- rewards: reward configs and reward wrappers
- callbacks: SB3 callbacks (viz, best-model saving)
- args: CLI parsing helpers
"""
