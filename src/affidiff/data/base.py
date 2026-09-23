from abc import abstractmethod

import polars as pl


class BaseDataLoader:
    @abstractmethod
    def load(self) -> pl.LazyFrame:
        pass
