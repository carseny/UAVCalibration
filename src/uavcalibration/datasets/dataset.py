from torch.utils.data import Dataset
import numpy as np
import cv2

from dataclasses import dataclass, field


@dataclass
class UAVData:
    uav_image: np.ndarray
    satellite_image: np.ndarray
    # (west, south, east, north) in lat/lon coordinates
    bbox: tuple[float, float, float, float] | None = None

    calibration_mat: np.ndarray = field(default_factory=lambda: np.eye(3))
    calibrated_shape: tuple[int, int] = field(init=False)  # (width, height)

    longitude: float | None = None  # in degrees
    latitude: float | None = None  # in degrees
    height: float | None = None  # in meters
    pitch: float | None = None  # in radians
    roll: float | None = None  # in radians
    yaw: float | None = None  # in radians
    focal_length: float = 0  # in pixels

    def __post_init__(self):
        h, w, *_ = self.uav_image.shape
        self.calibrated_shape = (w, h)
        # If focal length is not provided, set it to a reasonable default value
        if self.focal_length == 0:
            self.focal_length = max(h, w) * 1.5

    def apply_mat(self, H: np.ndarray):
        # Update the calibration matrix
        self.calibration_mat = H @ self.calibration_mat
        # Calculate the calibrated shape
        h, w, *_ = self.uav_image.shape
        corner_src = np.array(
            [
                [0, 0, 1],
                [w, 0, 1],
                [w, h, 1],
                [0, h, 1],
            ]
        )
        corner_dst = corner_src @ self.calibration_mat.T
        corner_dst = corner_dst[:, :2] / corner_dst[:, 2:3]
        # Adjust the calibration matrix to make sure all coordinates are positive
        coord_min = corner_dst.min(axis=0)
        coord_max = corner_dst.max(axis=0)
        self.calibrated_shape = tuple((coord_max - coord_min).astype(int))
        self.calibration_mat[0:2, 2] -= coord_min

    @property
    def calibrated_image(self):
        w, h = self.calibrated_shape
        return cv2.warpPerspective(self.uav_image, self.calibration_mat, (w, h))

    @property
    def calibration_mask(self):
        w, h = self.calibrated_shape
        mask = np.ones_like(self.uav_image)
        mask = cv2.warpPerspective(mask, self.calibration_mat, (w, h))
        mask = mask >= 0.5
        return mask


class UAVDataset(Dataset[UAVData]): ...
