from __future__ import annotations


def patch_old_gym_render_mode() -> None:
    """Compat: Shimmy expects the underlying Gym env to expose `render_mode`.

    gym-donkeycar is built on older OpenAI Gym envs that often do not define
    `render_mode`, which can crash Shimmy during wrapper init.
    """

    try:
        import gym as old_gym  # type: ignore
    except Exception:
        return

    if getattr(old_gym.make, "__name__", "") == "_make_with_render_mode":
        return

    original_make = old_gym.make

    def _make_with_render_mode(*args, **kwargs):
        env = original_make(*args, **kwargs)
        if not hasattr(env, "render_mode"):
            try:
                setattr(env, "render_mode", None)
            except Exception:
                pass
        return env

    old_gym.make = _make_with_render_mode


def patch_gym_donkeycar_stop_join() -> None:
    """Compat: gym-donkeycar can call `SDClient.stop()` from its own recv thread.

    In gym-donkeycar==1.3.1, `DonkeyUnitySimHandler.on_abort()` calls `self.client.stop()`.
    That abort callback runs inside the SDClient message-processing thread, and the
    default `stop()` implementation tries to `join()` the current thread, raising:

        RuntimeError: cannot join current thread

    This monkey patch makes `stop()` skip `join()` when called from the recv thread.
    """

    try:
        import threading

        import gym_donkeycar.core.client as client  # type: ignore
    except Exception:
        return

    sd = getattr(client, "SDClient", None)
    if sd is None:
        return

    original_stop = getattr(sd, "stop", None)
    if original_stop is None:
        return

    if getattr(original_stop, "__name__", "") == "_stop_no_join_current_thread":
        return

    def _stop_no_join_current_thread(self) -> None:  # type: ignore[no-untyped-def]
        self.do_process_msgs = False
        th = getattr(self, "th", None)
        if th is not None and th is not threading.current_thread():
            try:
                th.join()
            except Exception:
                pass
        s = getattr(self, "s", None)
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    sd.stop = _stop_no_join_current_thread
