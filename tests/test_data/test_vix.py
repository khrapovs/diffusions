import polars as pl

from affidiff.data.vix import VIX


class TestVIX:
    def test_load(self) -> None:
        """Test VIX data loading: lazy frame, collection, and columns."""
        vix = VIX()
        lazy_df = vix.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}.issubset(set(df.columns))

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        vix = VIX()
        df1 = vix.load().collect()
        df2 = vix.load().collect()
        assert df1.equals(df2)
