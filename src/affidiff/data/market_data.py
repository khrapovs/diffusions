import polars as pl

from affidiff.data.base import BaseDataLoader
from affidiff.data.spx import SPX
from affidiff.data.vix import VIX


class MarketData(BaseDataLoader):
    def load(self) -> pl.LazyFrame:
        """Load and merge VIX and SPX data on date, using SPX closing price only."""
        vix = VIX()
        spx = SPX()

        vix_df = vix.load().select("DATE", "CLOSE").rename({"CLOSE": "VIX"})
        spx_df = spx.load().select("DATE", "SPX")

        merged = vix_df.join(spx_df, on="DATE", how="inner").drop_nulls()
        return merged
