"""Registry of available models and their parameter classes."""

from affidiff.models.cir import CIR
from affidiff.models.ct import CentTend
from affidiff.models.gbm import GBM
from affidiff.models.heston import Heston
from affidiff.models.vasicek import Vasicek
from affidiff.parameters.cir import CIRparam
from affidiff.parameters.ct import CentTendParam
from affidiff.parameters.gbm import GBMparam
from affidiff.parameters.heston import HestonParam
from affidiff.parameters.vasicek import VasicekParam

MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "GBM": (GBM, GBMparam),
    "Vasicek": (Vasicek, VasicekParam),
    "CIR": (CIR, CIRparam),
    "Heston": (Heston, HestonParam),
    "CentTend": (CentTend, CentTendParam),
}
