from datetime import datetime, timedelta

import polars as pl

from affidiff.data.spx_high_freq import SPXHighFreq


class TestSPXHighFreq:
    @staticmethod
    def _build_loader() -> SPXHighFreq:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=5)
        return SPXHighFreq(start_time=start_time, end_time=end_time, interval="1h")

    def test_load(self) -> None:
        """Test SPXHighFreq data loading: lazy frame, collection, and columns."""
        spx = self._build_loader()
        lazy_df = spx.load()
        assert isinstance(lazy_df, pl.LazyFrame)

        df = lazy_df.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert {"DATE", "SPX"}.issubset(set(df.columns))

    def test_load_idempotent(self) -> None:
        """Test that multiple calls to load return equivalent data."""
        spx = self._build_loader()
        df1 = spx.load().collect()
        df2 = spx.load().collect()
        assert df1.equals(df2)
