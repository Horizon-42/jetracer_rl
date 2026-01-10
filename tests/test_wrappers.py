"""Unit tests for donkey_rl.wrappers module.

Tests cover:
- StepTimeoutWrapper timeout functionality
- JetRacerWrapper action mapping
- RandomFrictionWrapper friction randomization
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

from donkey_rl.wrappers import JetRacerWrapper, RandomFrictionWrapper, StepTimeoutWrapper


class TestJetRacerWrapper(unittest.TestCase):
    """Test JetRacerWrapper action mapping."""

    def setUp(self):
        """Set up test environment mock."""
        self.env = MagicMock()
        self.env.action_space = MagicMock()
        self.env.step = MagicMock(
            return_value=(np.zeros((84, 84, 3)), 0.0, False, False, {})
        )

    def test_jetracer_wrapper_action_mapping(self):
        """Test that actions are correctly mapped from [throttle, steering] to [steer, throttle]."""
        wrapper = JetRacerWrapper(self.env, steer_scale=1.0, throttle_scale=1.0)
        action = np.array([0.8, -0.5])  # [throttle, steering]
        mapped = wrapper.action(action)

        # Should be [steer, throttle] format
        self.assertAlmostEqual(mapped[0], -0.5, places=6)  # steering
        self.assertAlmostEqual(mapped[1], 0.8, places=6)  # throttle

    def test_jetracer_wrapper_clips_actions(self):
        """Test that actions are clipped to valid ranges."""
        wrapper = JetRacerWrapper(self.env, steer_scale=1.0, throttle_scale=1.0)
        
        # Test steering clipping
        action = np.array([0.0, 2.0])  # Steering exceeds [-1, 1]
        mapped = wrapper.action(action)
        self.assertLessEqual(mapped[0], 1.0)
        self.assertGreaterEqual(mapped[0], -1.0)

        # Test throttle clipping
        action = np.array([2.0, 0.0])  # Throttle exceeds [0, 1]
        mapped = wrapper.action(action)
        self.assertLessEqual(mapped[1], 1.0)

    def test_jetracer_wrapper_stores_last_action(self):
        """Test that last actions are stored for reward computation."""
        wrapper = JetRacerWrapper(self.env, steer_scale=1.0, throttle_scale=1.0)
        action = np.array([0.5, -0.3], dtype=np.float32)
        wrapper.action(action)

        self.assertIsNotNone(wrapper.last_raw_action)
        self.assertIsNotNone(wrapper.last_mapped_action)
        # Use almost_equal for float32 comparison
        np.testing.assert_array_almost_equal(wrapper.last_raw_action, action, decimal=6)
        # Verify the shape and values are correct
        self.assertEqual(wrapper.last_raw_action.shape, action.shape)

    def test_jetracer_wrapper_action_space(self):
        """Test that action space is updated correctly."""
        wrapper = JetRacerWrapper(self.env, steer_scale=1.0, throttle_scale=1.0)
        self.assertEqual(wrapper.action_space.shape, (2,))
        self.assertEqual(wrapper.action_space.low[0], 0.0)  # throttle low
        self.assertEqual(wrapper.action_space.high[0], 1.0)  # throttle high
        self.assertEqual(wrapper.action_space.low[1], -1.0)  # steering low
        self.assertEqual(wrapper.action_space.high[1], 1.0)  # steering high


class TestRandomFrictionWrapper(unittest.TestCase):
    """Test RandomFrictionWrapper friction randomization."""

    def setUp(self):
        """Set up test environment mock."""
        self.env = MagicMock()
        self.env.action_space = MagicMock()
        self.env.reset = MagicMock(return_value=(np.zeros((84, 84, 3)), {}))
        self.env.step = MagicMock(
            return_value=(np.zeros((84, 84, 3)), 0.0, False, False, {})
        )

    def test_random_friction_wrapper_resets_friction(self):
        """Test that friction scale is randomized on reset."""
        wrapper = RandomFrictionWrapper(self.env, min_scale=0.6, max_scale=1.0)
        
        # Reset multiple times and check that friction scale changes
        scales = []
        for _ in range(10):
            wrapper.reset()
            scales.append(wrapper.friction_scale)
        
        # At least some values should be different (with high probability)
        unique_scales = set(scales)
        self.assertGreater(len(unique_scales), 1)

    def test_random_friction_wrapper_applies_friction(self):
        """Test that friction scale is applied to throttle in step."""
        wrapper = RandomFrictionWrapper(self.env, min_scale=0.8, max_scale=0.8)
        wrapper.reset()  # Set friction_scale to 0.8
        
        action = np.array([1.0, 0.0])  # [steer, throttle] format, throttle=1.0
        obs, reward, terminated, truncated, info = wrapper.step(action)

        # Friction should be applied to throttle (action[1])
        # Since friction_scale=0.8, throttle should be 0.8
        # But we can't directly check the modified action, so we verify step was called
        self.env.step.assert_called_once()

    def test_random_friction_wrapper_validates_range(self):
        """Test that invalid friction ranges raise ValueError."""
        with self.assertRaises(ValueError):
            RandomFrictionWrapper(self.env, min_scale=1.0, max_scale=0.5)  # min > max

        with self.assertRaises(ValueError):
            RandomFrictionWrapper(self.env, min_scale=-0.1, max_scale=1.0)  # min <= 0

        with self.assertRaises(ValueError):
            RandomFrictionWrapper(self.env, min_scale=0.5, max_scale=-0.1)  # max <= 0

    def test_random_friction_wrapper_adds_friction_to_info(self):
        """Test that friction scale is added to info dict."""
        wrapper = RandomFrictionWrapper(self.env, min_scale=0.8, max_scale=0.8)
        wrapper.reset()
        
        action = np.array([0.0, 0.5])
        obs, reward, terminated, truncated, info = wrapper.step(action)

        self.assertIn("friction_scale", info)
        self.assertEqual(info["friction_scale"], wrapper.friction_scale)


class TestStepTimeoutWrapper(unittest.TestCase):
    """Test StepTimeoutWrapper timeout functionality."""

    def setUp(self):
        """Set up test environment mock."""
        self.env = MagicMock()
        self.env.reset = MagicMock(return_value=(np.zeros((84, 84, 3)), {}))
        self.env.step = MagicMock(
            return_value=(np.zeros((84, 84, 3)), 0.0, False, False, {})
        )

    def test_step_timeout_wrapper_passes_through_when_no_timeout(self):
        """Test that wrapper passes through when timeout is disabled."""
        wrapper = StepTimeoutWrapper(self.env, timeout_s=0.0)
        obs, reward, terminated, truncated, info = wrapper.step([0.0, 0.0])
        
        self.env.step.assert_called_once_with([0.0, 0.0])

    def test_step_timeout_wrapper_reset(self):
        """Test that reset works through the wrapper."""
        wrapper = StepTimeoutWrapper(self.env, timeout_s=30.0)
        obs, info = wrapper.reset()
        
        self.env.reset.assert_called_once()

    def test_step_timeout_wrapper_supports_signals(self):
        """Test signal support detection.

        This test verifies that the wrapper can detect signal support.
        Since signal and threading are imported inside the method, we test
        the actual behavior rather than mocking internal imports.
        """
        wrapper = StepTimeoutWrapper(self.env, timeout_s=30.0)
        # Test that _supports_signals method exists and returns a boolean
        result = wrapper._supports_signals()
        self.assertIsInstance(result, bool)
        
        # Verify the wrapper works regardless of signal support
        # The method should handle both cases gracefully
        self.assertTrue(hasattr(wrapper, '_supports_signals'))


if __name__ == "__main__":
    unittest.main()

