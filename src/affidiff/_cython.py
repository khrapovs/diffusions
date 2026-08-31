"""Utilities for loading optional Cython extensions."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

_simulate_cache: Callable | None | str = "uninitialized"  # Use sentinel to detect first call


def get_cython_simulate() -> Callable | None:
    """Load the Cython simulate function if available.

    Uses lazy loading with caching to avoid repeated import attempts.
    First call loads and caches the result, subsequent calls return the cached value.

    Returns
    -------
    Callable or None
        The cython simulate function, or None if not available.
    """
    global _simulate_cache

    # Return cached result if already loaded
    if _simulate_cache != "uninitialized":
        return _simulate_cache  # type: ignore

    # First call - attempt to load
    try:
        spec = importlib.util.find_spec("affidiff.simulate")
        if spec is None:
            _simulate_cache = None
            return None
        from affidiff.simulate import simulate  # type: ignore

        _simulate_cache = simulate
        return simulate
    except ImportError, ModuleNotFoundError:
        _simulate_cache = None
        return None
