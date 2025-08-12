from dataclasses import dataclass

from torch.utils.data import Dataset
import numpy as np


@dataclass
class UAVData:
    uav_image: np.ndarray
    satellite_image: np.ndarray
    satellite_transform: np.ndarray
    satellite_crs: str

    longitude: float | None = None  # in degrees
    latitude: float | None = None  # in degrees
    height: float | None = None  # in meters
    yaw: float | None = None  # in radians
    pitch: float | None = None  # in radians
    roll: float | None = None  # in radians
    focal_length: float = 0  # in pixels

    def __post_init__(self):
        h, w, *_ = self.uav_image.shape
        self.calibrated_shape = (w, h)
        # If focal length is not provided, set it to a reasonable default value
        if self.focal_length == 0:
            self.focal_length = max(h, w) * 1.5


class UAVDataset(Dataset[UAVData]): ...
