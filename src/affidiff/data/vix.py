import polars as pl


class VIX:
    _URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def load(self) -> pl.LazyFrame:
        """Load VIX historical data as a lazy polars dataframe."""
        return pl.scan_csv(self._URL)
