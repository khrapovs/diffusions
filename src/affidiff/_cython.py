from __future__ import annotations

import importlib.util
from collections.abc import Callable


def get_cython_simulate() -> Callable | None:
    """Load the Cython simulate function if available.

    Returns
    -------
    Callable or None
        The cython simulate function, or None if not available.
    """
    try:
        spec = importlib.util.find_spec("affidiff.simulate")
        if spec is None:
            return None
        from affidiff.simulate import simulate  # type: ignore

        return simulate
    except ImportError, ModuleNotFoundError:
        return None
