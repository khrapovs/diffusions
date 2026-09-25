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
        df = df.reset_index()
        df["Datetime"] = df["Datetime"].dt.tz_convert(None)
        # Build columns manually to avoid the optional pyarrow dependency required by pl.from_pandas.
        data = {col: df[col].to_numpy() for col in df.columns}
        data["Datetime"] = data["Datetime"].astype("datetime64[us]")
        return pl.DataFrame(data).rename({"Datetime": "DATE", "Close": "SPX"}).lazy()
