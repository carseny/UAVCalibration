from ..types import ModelOutput

import numpy as np
import cv2

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MatchingMethod",
    "MatchResult",
]


class MatchingMethod(Enum):
    SIFT = "sift"
    LIGHTGLUE = "lightglue"

    HOMOGRAPHY = "homography"


@dataclass
class MatchResult:
    methed: MatchingMethod
    kpts0: np.ndarray
    kpts1: np.ndarray
    scores: np.ndarray
