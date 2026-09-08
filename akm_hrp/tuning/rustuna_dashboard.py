from __future__ import annotations

import threading
import time
from typing import List, Dict, Optional

import plotly.graph_objects as go
import plotly.io as pio


class RustunaRealtimeDashboard:
    def __init__(self):
        self._lock = threading.Lock()
        self._history: List[Dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, refresh_seconds: float = 2.0) -> None:
        if self._running:
            return

        self._running = True

        def _loop():
            while self._running:
                self._render()
                time.sleep(refresh_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def record(
        self,
        stage: str,
        completed: int,
        stage_total: int,
        value: float | None,
        best_value: float | None,
    ) -> None:
        with self._lock:
            self._history.append(
                {
                    "stage": stage,
                    "completed": completed,
                    "stage_total": stage_total,
                    "value": value,
                    "best": best_value,
                }
            )

    def _render(self) -> None:
        with self._lock:
            if not self._history:
                return

            overlay_x = []
            overlay_y = []
            baseline_x = []
            baseline_y = []

            for h in self._history:
                if h["value"] is None:
                    continue
                if h["stage"] == "overlay":
                    overlay_x.append(h["completed"])
                    overlay_y.append(h["value"])
                elif h["stage"] == "baseline":
                    baseline_x.append(h["completed"])
                    baseline_y.append(h["value"])

        fig = go.Figure()
        if overlay_x:
            fig.add_trace(
                go.Scatter(
                    x=overlay_x,
                    y=overlay_y,
                    mode="lines+markers",
                    name="Overlay",
                    line=dict(color="royalblue"),
                )
            )
        if baseline_x:
            fig.add_trace(
                go.Scatter(
                    x=baseline_x,
                    y=baseline_y,
                    mode="lines+markers",
                    name="Baseline",
                    line=dict(color="firebrick"),
                )
            )

        fig.update_layout(
            title="Rustuna Real-Time Progress",
            xaxis_title="Completed trials (per stage)",
            yaxis_title="Score",
            template="plotly_white",
        )

        pio.show(fig)

