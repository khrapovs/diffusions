from io import BytesIO

import polars as pl
import requests

from affidiff.data.base import BaseDataLoader


class VIX(BaseDataLoader):
    _URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def load(self) -> pl.LazyFrame:
        """Load VIX historical data as a lazy polars dataframe with date conversion."""
        response = requests.get(self._URL)
        response.raise_for_status()
        return pl.read_csv(BytesIO(response.content)).with_columns(pl.col("DATE").str.to_date("%m/%d/%Y")).lazy()
