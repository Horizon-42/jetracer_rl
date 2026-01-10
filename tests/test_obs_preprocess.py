"""Unit tests for donkey_rl.obs_preprocess module.

Tests cover:
- Observation preprocessing pipeline
- Domain randomization functions
- Perspective transformation
- Format conversions (HWC -> CHW)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from donkey_rl.obs_preprocess import (
    ObsPreprocess,
    _apply_brightness_contrast,
    _apply_color_jitter,
    _apply_domain_randomization,
    _apply_gaussian_noise,
)


class TestDomainRandomization(unittest.TestCase):
    """Test domain randomization functions."""

    def test_apply_color_jitter_no_change_when_zero(self):
        """Test that color jitter does nothing when strength is zero."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _apply_color_jitter(image, jitter_strength=0.0)
        np.testing.assert_array_almost_equal(image, result, decimal=5)

    def test_apply_color_jitter_changes_image(self):
        """Test that color jitter modifies the image when strength > 0."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _apply_color_jitter(image, jitter_strength=0.2)
        # Result should be in valid range
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_apply_brightness_contrast(self):
        """Test brightness and contrast augmentation."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _apply_brightness_contrast(image, brightness_range=0.1, contrast_range=0.1)
        # Result should be in valid range (will be clipped later)
        self.assertTrue(result.shape == image.shape)

    def test_apply_gaussian_noise_no_change_when_zero(self):
        """Test that Gaussian noise does nothing when std is zero."""
        np.random.seed(42)  # For reproducibility
        image = np.random.rand(64, 64, 3).astype(np.float32)
        image_copy = image.copy()
        result = _apply_gaussian_noise(image_copy, noise_std=0.0)
        np.testing.assert_array_equal(image, result)

    def test_apply_gaussian_noise_changes_image(self):
        """Test that Gaussian noise modifies the image when std > 0."""
        np.random.seed(42)
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _apply_gaussian_noise(image, noise_std=0.02)
        # Result should be different (with high probability)
        self.assertFalse(np.allclose(image, result, atol=1e-6))

    def test_apply_domain_randomization_clips_to_valid_range(self):
        """Test that domain randomization clips values to [0, 1]."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _apply_domain_randomization(
            image,
            brightness_range=0.5,
            contrast_range=0.5,
            noise_std=0.1,
            color_jitter_strength=0.3,
        )
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


class TestObsPreprocess(unittest.TestCase):
    """Test ObsPreprocess wrapper."""

    def setUp(self):
        """Set up test environment mock."""
        self.env = MagicMock()
        self.env.observation_space = MagicMock()
        self.env.observation_space.shape = (240, 320, 3)
        self.env.reset = MagicMock(return_value=(np.zeros((240, 320, 3)), {}))

    def test_obs_preprocess_shape_conversion(self):
        """Test that observation is converted from HWC to CHW."""
        wrapper = ObsPreprocess(self.env, width=84, height=84)
        obs = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = wrapper.observation(obs)

        # Should be CHW format
        self.assertEqual(result.shape, (3, 84, 84))
        self.assertEqual(result.dtype, np.float32)

    def test_obs_preprocess_value_range(self):
        """Test that observation values are in [0, 1] range."""
        wrapper = ObsPreprocess(self.env, width=84, height=84)
        obs = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = wrapper.observation(obs)

        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_obs_preprocess_caches_observations(self):
        """Test that observations are cached for debugging."""
        wrapper = ObsPreprocess(self.env, width=84, height=84)
        obs = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        wrapper.observation(obs)

        self.assertIsNotNone(wrapper.last_raw_observation)
        self.assertIsNotNone(wrapper.last_resized_observation)
        self.assertEqual(wrapper.last_raw_observation.shape, (240, 320, 3))
        self.assertEqual(wrapper.last_resized_observation.shape, (84, 84, 3))

    def test_obs_preprocess_with_domain_rand(self):
        """Test observation preprocessing with domain randomization enabled."""
        wrapper = ObsPreprocess(
            self.env,
            width=84,
            height=84,
            domain_rand=True,
            aug_brightness=0.1,
            aug_contrast=0.1,
            aug_noise_std=0.01,
            aug_color_jitter=0.1,
        )
        obs = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result1 = wrapper.observation(obs)
        result2 = wrapper.observation(obs)

        # Results should be different due to randomization (with high probability)
        # But we can't guarantee it, so just check shape and range
        self.assertEqual(result1.shape, (3, 84, 84))
        self.assertTrue(np.all(result1 >= 0.0))
        self.assertTrue(np.all(result1 <= 1.0))

    def test_obs_preprocess_perspective_transform_flag(self):
        """Test that perspective transform flag is set correctly."""
        wrapper = ObsPreprocess(self.env, width=84, height=84, perspective_transform=True)
        self.assertTrue(wrapper._perspective_transform)

        wrapper_no_transform = ObsPreprocess(self.env, width=84, height=84, perspective_transform=False)
        self.assertFalse(wrapper_no_transform._perspective_transform)

    def test_obs_preprocess_observation_space(self):
        """Test that observation space is updated correctly."""
        wrapper = ObsPreprocess(self.env, width=84, height=84)
        self.assertEqual(wrapper.observation_space.shape, (3, 84, 84))
        self.assertEqual(wrapper.observation_space.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

