from enum import StrEnum


class Measure(StrEnum):
    P = "P"  # physical
    Q = "Q"  # risk-neutral
    PQ = "PQ"  # both


class Subset(StrEnum):
    all = "all"  # all parameters, including those related to returns
    vol = "vol"  # only those related to volatility
