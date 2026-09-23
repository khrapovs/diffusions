import polars as pl

from affidiff.data.base import BaseDataLoader


class SPX(BaseDataLoader):
    _URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv"

    def load(self) -> pl.LazyFrame:
        """Load S&P 500 historical data as a lazy polars dataframe with date conversion."""
        return pl.scan_csv(self._URL).with_columns(pl.col("DATE").str.to_date("%m/%d/%Y"))
