import polars as pl

from affidiff.data.spx import SPX


class TestSPX:
    def test_load_returns_lazy_frame(self) -> None:
        """Test that load returns a polars LazyFrame."""
        spx = SPX()
        result = spx.load()
        assert isinstance(result, pl.LazyFrame)

    def test_load_can_be_collected(self) -> None:
        """Test that the LazyFrame can be collected into a DataFrame."""
        spx = SPX()
        lazy_df = spx.load()
        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_has_expected_columns(self) -> None:
        """Test that the loaded data contains expected SPX columns."""
        spx = SPX()
        lazy_df = spx.load()
        df = lazy_df.collect()

        expected_columns = {"DATE", "SPX"}
        actual_columns = set(df.columns)
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        spx = SPX()
        df1 = spx.load().collect()
        df2 = spx.load().collect()
        assert df1.equals(df2)
