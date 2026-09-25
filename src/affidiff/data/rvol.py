from datetime import datetime

import polars as pl

from affidiff.data.base import BaseDataLoader
from affidiff.data.spx_high_freq import SPXHighFreq


class RVOL(BaseDataLoader):
    def __init__(self, *, start_time: datetime, end_time: datetime, interval: str) -> None:
        self._spx_loader = SPXHighFreq(start_time=start_time, end_time=end_time, interval=interval)

    def load(self) -> pl.LazyFrame:
        """Compute daily realized volatility from high-frequency SPX log returns.

        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        rvol = np.sum(log_returns ** 2)

        Log returns are computed within each calendar day, so the overnight
        return between the last observation of one day and the first
        observation of the next day is excluded from the sum.
        """
        return (
            self._spx_loader.load()
            .sort("DATE")
            .with_columns(pl.col("DATE").dt.date().alias("DAY"))
            .with_columns(pl.col("SPX").log().diff().over("DAY").alias("LOG_RETURN"))
            .group_by("DAY")
            .agg((pl.col("LOG_RETURN") ** 2).sum().alias("RVOL"))
            .rename({"DAY": "DATE"})
            .sort("DATE")
        )
