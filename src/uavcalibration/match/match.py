from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "MatchMethod",
    "MatchResult",
]


class MatchMethod(Enum):
    SIFT = "sift"
    LIGHTGLUE = "lightglue"

    HOMOGRAPHY = "homography"


@dataclass
class MatchResult:
    methed: MatchMethod
    kpts0: np.ndarray
    kpts1: np.ndarray
    scores: np.ndarray
