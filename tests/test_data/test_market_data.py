import polars as pl

from affidiff.data.market_data import MarketData


class TestMarketData:
    def test_load(self) -> None:
        """Test MarketData: lazy frame, collection, columns, date type, no nulls."""
        market_data = MarketData()
        lazy_df = market_data.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "VIX", "SPX"}.issubset(set(df.columns))
        assert df.schema["DATE"] == pl.Date
        assert df.null_count().sum_horizontal()[0] == 0
        assert df["DATE"].min() is not None
        assert df["DATE"].max() is not None
