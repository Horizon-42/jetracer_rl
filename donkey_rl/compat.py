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
