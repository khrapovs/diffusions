import polars as pl


class SPX:
    _URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv"

    def load(self) -> pl.LazyFrame:
        """Load S&P 500 historical data as a lazy polars dataframe."""
        return pl.scan_csv(self._URL)
