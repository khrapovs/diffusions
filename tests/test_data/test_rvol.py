from datetime import datetime, timedelta

import polars as pl

from affidiff.data.rvol import RVOL


class TestRVOL:
    @staticmethod
    def _build_loader() -> RVOL:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)
        return RVOL(start_time=start_time, end_time=end_time, interval="1h")

    def test_load(self) -> None:
        """Test RVOL data loading: lazy frame, collection, and columns."""
        rvol = self._build_loader()
        lazy_df = rvol.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "RVOL"}.issubset(set(df.columns))
        assert df["RVOL"].to_numpy().min() >= 0

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        rvol = self._build_loader()
        df1 = rvol.load().collect()
        df2 = rvol.load().collect()
        assert df1.equals(df2)
