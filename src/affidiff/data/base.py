from abc import ABC, abstractmethod

import polars as pl


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> pl.LazyFrame:
        pass
