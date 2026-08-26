"""Optional Omni.UI command panel for RSL-RL velocity-conditioned tasks."""

from __future__ import annotations

import threading

import numpy as np


class LiveSE2Command:
    """Thread-safe ``[vx, vy, wz]`` state shared by UI and simulation."""

    def __init__(self, command: tuple[float, float, float], bounds: tuple[tuple[float, float], ...]):
        if len(command) != 3 or len(bounds) != 3:
            raise ValueError("A live SE(2) command requires three values and three bounds.")
        self._bounds = tuple((float(low), float(high)) for low, high in bounds)
        self._command = np.asarray(
            [np.clip(float(value), low, high) for value, (low, high) in zip(command, self._bounds)],
            dtype=np.float32,
        )
        self._lock = threading.Lock()

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            return tuple(float(value) for value in self._command)

    def set_component(self, index: int, value: float) -> None:
        low, high = self._bounds[index]
        with self._lock:
            self._command[index] = np.clip(float(value), low, high)


def build_live_command_window(
    state: LiveSE2Command,
    *,
    title: str = "Solo12 live command",
    subtitle: str = "Edit vx, vy and yaw rate while the policy is running.",
):
    """Create an Omni.UI floating panel and keep callback handles alive."""

    import omni.ui as ui

    window = ui.Window(title, width=500, height=245, visible=True)
    keepalive = []
    labels = (("vx [m/s]", 0), ("vy [m/s]", 1), ("wz [rad/s]", 2))
    with window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label(title, height=24)
            ui.Label(subtitle, height=18)

            for label, index in labels:
                low, high = state._bounds[index]
                with ui.HStack(spacing=8, height=30):
                    ui.Label(label, width=145)
                    model = ui.SimpleFloatModel(state.get()[index], min=low, max=high)
                    ui.FloatField(model, width=115)
                    ui.FloatSlider(model, min=low, max=high, width=190)
                    callback = lambda changed, i=index: state.set_component(i, changed.get_value_as_float())
                    if hasattr(model, "subscribe_value_changed_fn"):
                        keepalive.append(model.subscribe_value_changed_fn(callback))
                    else:
                        model.add_value_changed_fn(callback)
                        keepalive.append(callback)

            ui.Label("The sliders are clipped to the bound training envelope.", height=18)
    return window, keepalive
