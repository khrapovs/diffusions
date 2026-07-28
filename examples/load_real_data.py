#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load real market data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datastorage.cboe import load_vix_spx  # type: ignore
from datastorage.oxfordman import load_realized_vol  # type: ignore


def load_data() -> tuple[Any, Any]:
    """Load and process real market data.

    Returns
    -------
    tuple
        Tuple of (ret, rvar) arrays

    """
    realized_vol = load_realized_vol()
    vix_spx = load_vix_spx()
    data = pd.merge(realized_vol, vix_spx, left_index=True, right_index=True)
    data["logR"] = data["SPX"].apply(np.log).diff(1)
    data.dropna(inplace=True)

    data_arr = data[["logR", "RV"]].values.T
    ret, rvar = data_arr
    rvar = (rvar / 100) ** 2

    return (ret, rvar)


if __name__ == "__main__":
    pass
