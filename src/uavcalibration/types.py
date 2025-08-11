from typing import Any

import numpy as np

__all__ = ["Shape", "ImageMat"]

type Width = int
type Height = int
type Shape = tuple[Width, Height]
type ImageMat = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]
