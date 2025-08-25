from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Shape", "ImageMat", "NDArray"]

type Width = int
type Height = int
type Shape = tuple[Width, Height]
type ImageMat = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
