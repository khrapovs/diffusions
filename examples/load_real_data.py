#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load real market data


"""
from __future__ import print_function, division

import numpy as np
import pandas as pd

from datastorage.oxfordman import load_realized_vol
from datastorage.cboe import load_vix_spx


def load_data():

    realized_vol = load_realized_vol()
    vix_spx = load_vix_spx()
    data = pd.merge(realized_vol, vix_spx, left_index=True, right_index=True)
    data['logR'] = data['SPX'].apply(np.log).diff(1)
    data.dropna(inplace=True)

    data = data[['logR', 'RV']].values.T
    ret, rvar = data
    rvar = (rvar / 100) ** 2

    return (ret, rvar)


if __name__ == '__main__':

    pass