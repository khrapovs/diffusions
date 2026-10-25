# Purpose

`src/affidiff/data/` provides loaders that fetch market data from external sources and return it as `polars`
lazy frames for use in estimation and calibration.

# Ownership

- `base.py` — `BaseDataLoader` abstract base class declaring the `load() -> pl.LazyFrame` contract.
- `spx.py` — `SPX`, daily S&P 500 history from the CBOE CSV endpoint.
- `vix.py` — `VIX`, daily VIX history from the CBOE CSV endpoint.
- `spx_high_freq.py` — `SPXHighFreq`, intraday `^GSPC` bars via `yfinance`.
- `market_data.py` — `MarketData`, inner join of VIX and SPX closes on `DATE`.
- `rvol.py` — `RVOL`, daily realized variance computed from intraday SPX log returns.

# Local Contracts

- Every loader subclasses `BaseDataLoader` and exposes exactly one public method, `load()`, returning a
  `pl.LazyFrame`; constructor arguments are keyword-only and stored privately.
- Column naming is uppercase and stable across loaders: `DATE`, `SPX`, `VIX`, `RVOL`.
- Endpoint URLs and other loader configuration are private class attributes (`_URL`), not module-level constants.
- Loaders perform network I/O; they must not cache to disk or mutate repository state.

# Work Guidance

- Raise on failed HTTP responses (`response.raise_for_status()`) rather than returning partial frames.
- Keep transformations lazy; return `pl.LazyFrame` and let callers decide when to collect.
- Realized-measure computations must stay within calendar days so overnight returns are excluded.

# Verification

- Run data tests: `uv run pytest tests/test_data`
- Run pre-commit hooks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

None (leaf boundary).
