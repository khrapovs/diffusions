import polars as pl

from affidiff.data.vix import VIX


class TestVIX:
    def test_load_returns_lazy_frame(self) -> None:
        """Test that load returns a polars LazyFrame."""
        vix = VIX()
        result = vix.load()
        assert isinstance(result, pl.LazyFrame)

    def test_load_can_be_collected(self) -> None:
        """Test that the LazyFrame can be collected into a DataFrame."""
        vix = VIX()
        lazy_df = vix.load()
        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_load_has_expected_columns(self) -> None:
        """Test that the loaded data contains expected VIX columns."""
        vix = VIX()
        lazy_df = vix.load()
        df = lazy_df.collect()

        expected_columns = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}
        actual_columns = set(df.columns)
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        vix = VIX()
        df1 = vix.load().collect()
        df2 = vix.load().collect()
        assert df1.equals(df2)
