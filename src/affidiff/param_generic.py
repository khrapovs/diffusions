"""Generic parameter class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


class GenericParam(ABC):
    """Generic parameter storage. Must be overriden.

    Attributes
    ----------
    measure : str
        Probability measure.
    """

    measure: str = "P"

    def __init__(self) -> None:
        """Initialize class."""
        self.measure = "P"

    def is_valid(self) -> bool:
        """Check whether parameters are valid.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return True

    @abstractmethod
    def update_ajd(self) -> None:
        """Update AJD representation."""
        raise NotImplementedError("Must be overridden")

    @classmethod
    @abstractmethod
    def from_theta(cls, theta: np.ndarray | Sequence[float]) -> Self:
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        raise NotImplementedError("Must be overridden")

    @abstractmethod
    def update(self, theta: np.ndarray | Sequence[float], subset: str = "all", measure: str = "P") -> None:  # noqa: PLR0917
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']
        measure : str
            Either physical measure (P), or risk-neutral (Q)

        """
        raise NotImplementedError("Must be overridden")

    @staticmethod
    @abstractmethod
    def get_model_name() -> str:
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        raise NotImplementedError("Must be overridden")

    @staticmethod
    @abstractmethod
    def get_names() -> list[str]:
        """Return parameter names.

        Returns
        -------
        list of str
            Parameter names

        """
        raise NotImplementedError("Must be overridden")

    @abstractmethod
    def get_theta(self) -> np.ndarray:
        """Return vector of parameters.

        Returns
        -------
        array
            Parameter vector

        """
        raise NotImplementedError("Must be overridden")

    @staticmethod
    def get_bounds(subset: str = "all", measure: str = "PQ") -> list[tuple[float | None, float | None]] | None:  # noqa: PLR0917, ARG004
        """Get parameter bounds.

        Returns
        -------
        list of tuples
            Parameter bounds

        """
        return None

    def get_constraints(self) -> tuple[dict[str, Any], ...] | list[dict[str, Any]] | tuple[()]:
        """Get parameter constraints.

        Returns
        -------
        dict or sequence of dict
            Equality and inequality constraints. See scipy.optimize.minimize

        """
        return ()

    def __str__(self) -> str:
        """Return string representation."""
        show = self.get_model_name() + " parameters under " + self.measure
        if self.is_valid():
            show += " (valid)"
        else:
            show += " (not valid)"
        show += ":\n"
        table = pd.DataFrame({"theta": self.get_theta()}, index=self.get_names())
        tb_str = table.to_string(float_format=lambda x: "%.4f" % x)
        width = len(tb_str) // (table.shape[0] + 1)
        show += width * "-" + "\n"
        show += tb_str
        show += "\n" + width * "-"
        return show

    def __repr__(self) -> str:
        """Return string representation."""
        return self.__str__()
