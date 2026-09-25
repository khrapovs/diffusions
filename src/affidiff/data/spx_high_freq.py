from datetime import datetime

import polars as pl
from yfinance import Ticker

from affidiff.data.base import BaseDataLoader


class SPXHighFreq(BaseDataLoader):
    def __init__(self, *, start_time: datetime, end_time: datetime, interval: str) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._interval = interval

    def load(self) -> pl.LazyFrame:
        """Load high-frequency S&P 500 historical data as a lazy polars dataframe."""
        t = Ticker("^GSPC")
        df = t.history(start=self._start_time, end=self._end_time, interval=self._interval)
        return (
            pl.LazyFrame(df.reset_index())
            .rename({"Datetime": "DATE", "Close": "SPX"})
            .with_columns(pl.col("DATE").dt.convert_time_zone("UTC"))
        )
