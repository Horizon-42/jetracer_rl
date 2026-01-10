"""Compatibility patches for gym/gymnasium/shimmy interop.

This module provides monkey patches to fix compatibility issues between:
- Old OpenAI Gym (gym-donkeycar dependency)
- Gymnasium (newer standard)
- Shimmy (Gymnasium compatibility layer)

These patches are applied at environment creation time to ensure smooth
interoperation between the different library versions.
"""

from __future__ import annotations


def patch_old_gym_render_mode() -> None:
    """Patch old Gym's make() to always set render_mode attribute.

    Problem:
        Shimmy (Gymnasium compatibility layer) expects all environments
        to expose a `render_mode` attribute. However, gym-donkeycar is built
        on older OpenAI Gym versions that often don't define this attribute,
        causing crashes during Shimmy wrapper initialization.

    Solution:
        Monkey-patch `gym.make()` to automatically set `render_mode=None`
        on any created environment that lacks this attribute.

    Notes:
        - Safe to call multiple times (idempotent)
        - Only patches if old gym is available
        - Silently fails if patching is not possible
    """
    try:
        import gym as old_gym  # type: ignore
    except ImportError:
        # Old gym not available, nothing to patch
        return

    # Check if already patched to avoid double-patching
    if getattr(old_gym.make, "__name__", "") == "_make_with_render_mode":
        return

    original_make = old_gym.make

    def _make_with_render_mode(*args, **kwargs):
        """Wrapper that ensures render_mode attribute exists."""
        env = original_make(*args, **kwargs)
        if not hasattr(env, "render_mode"):
            try:
                setattr(env, "render_mode", None)
            except (AttributeError, TypeError):
                # Some envs may prevent attribute setting, ignore gracefully
                pass
        return env

    old_gym.make = _make_with_render_mode


def patch_gym_donkeycar_stop_join() -> None:
    """Patch gym-donkeycar's SDClient.stop() to avoid thread join errors.

    Problem:
        In gym-donkeycar==1.3.1, when `DonkeyUnitySimHandler.on_abort()` is
        called, it invokes `self.client.stop()`. This abort callback runs
        inside the SDClient's message-processing thread. The default `stop()`
        implementation attempts to `join()` the current thread, which raises:

            RuntimeError: cannot join current thread

    Solution:
        Replace `SDClient.stop()` with a version that checks if we're trying
        to join the current thread before attempting the join operation.

    Notes:
        - Safe to call multiple times (idempotent)
        - Only patches if gym-donkeycar is available
        - Silently fails if patching is not possible
    """
    try:
        import threading
        import gym_donkeycar.core.client as client  # type: ignore
    except ImportError:
        # gym-donkeycar not available, nothing to patch
        return

    # Get the SDClient class if available
    sd_client_class = getattr(client, "SDClient", None)
    if sd_client_class is None:
        return

    # Get the original stop method
    original_stop = getattr(sd_client_class, "stop", None)
    if original_stop is None:
        return

    # Check if already patched
    if getattr(original_stop, "__name__", "") == "_stop_no_join_current_thread":
        return

    def _stop_no_join_current_thread(self) -> None:  # type: ignore[no-untyped-def]
        """Patched stop() that avoids joining the current thread.

        This implementation:
        1. Sets the stop flag to stop message processing
        2. Safely joins the message thread (only if it's not the current thread)
        3. Closes the socket connection
        """
        # Stop message processing loop
        self.do_process_msgs = False

        # Join the message thread only if it's not the current thread
        msg_thread = getattr(self, "th", None)
        if msg_thread is not None and msg_thread is not threading.current_thread():
            try:
                msg_thread.join()
            except (RuntimeError, AttributeError):
                # Thread already stopped or other error, ignore
                pass

        # Close the socket if it exists
        socket_obj = getattr(self, "s", None)
        if socket_obj is not None:
            try:
                socket_obj.close()
            except (OSError, AttributeError):
                # Socket already closed or other error, ignore
                pass

    sd_client_class.stop = _stop_no_join_current_thread
