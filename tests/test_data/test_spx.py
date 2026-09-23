import polars as pl

from affidiff.data.spx import SPX


class TestSPX:
    def test_load(self) -> None:
        """Test SPX data loading: lazy frame, collection, and columns."""
        spx = SPX()
        lazy_df = spx.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "SPX"}.issubset(set(df.columns))

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        spx = SPX()
        df1 = spx.load().collect()
        df2 = spx.load().collect()
        assert df1.equals(df2)
