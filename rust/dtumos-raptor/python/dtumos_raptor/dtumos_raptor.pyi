"""Type stubs for dtumos_raptor native module."""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

class DtumosRaptor:
    """Rust RAPTOR transit routing engine.

    Provides native RAPTOR routing via PyO3 bindings.
    Data is loaded from GTFS and can be hot-reloaded via arc-swap.
    """

    def __new__(cls, num_threads: Optional[int] = None) -> "DtumosRaptor": ...
    def load_gtfs(self, gtfs_dir: str) -> None:
        """Load GTFS data and build TransitData."""
        ...
    def reload_gtfs(self, gtfs_dir: str) -> dict[str, Any]:
        """Hot-reload GTFS: build new TransitData and atomically swap.

        Returns dict with build_time_ms, stops, routes, trips.
        """
        ...
    def update_config(self, config: dict[str, Any]) -> None:
        """Update cost/routing configuration from a Python dict."""
        ...
    def route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        departure_time_secs: int,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Route a single OD pair. Returns OTP-compatible JSON dict."""
        ...
    def route_batch(
        self,
        from_lats: NDArray[np.float64],
        from_lons: NDArray[np.float64],
        to_lats: NDArray[np.float64],
        to_lons: NDArray[np.float64],
        departure_times: NDArray[np.uint32],
        mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Batch routing with numpy arrays. Releases GIL for parallel execution."""
        ...
    def stats(self) -> dict[str, Any]:
        """Get engine statistics (loaded, stops, routes, trips, etc.)."""
        ...
