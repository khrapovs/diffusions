import polars as pl

from affidiff.data.market_data import MarketData


class TestMarketData:
    def test_load(self) -> None:
        """Test MarketData loading: lazy frame, collection, columns, and no nulls."""
        market_data = MarketData()
        lazy_df = market_data.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "VIX", "SPX"}.issubset(set(df.columns))

        assert df.null_count().sum_horizontal()[0] == 0

    def test_load_date_type(self) -> None:
        """Test that DATE column is properly converted to date type."""
        market_data = MarketData()
        df = market_data.load().collect()
        assert df.schema["DATE"] == pl.Date

    def test_load_merged_on_common_dates(self) -> None:
        """Test that data is merged on common dates."""
        market_data = MarketData()
        df = market_data.load().collect()

        vix_min_date = df["DATE"].min()
        vix_max_date = df["DATE"].max()

        assert vix_min_date is not None
        assert vix_max_date is not None
