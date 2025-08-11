import numpy as np
from numpy.typing import NDArray
import cv2
from pyproj import Transformer, Geod

from .types import *

__all__ = ["Transform", "PixelTransform", "CRSTransform"]


class PixelTransform:
    """Pixel to Pixel transform"""

    mat: NDArray[np.floating]  # 3x3 xy1->x'y'1
    src_shape: Shape | None
    dst_shape: Shape | None

    def __init__(
        self,
        mat: NDArray[np.floating] | None = None,
        src_shape: Shape | None = None,
        dst_shape: Shape | None = None,
    ):
        if mat is None:
            mat = np.eye(3)
        self.mat = mat
        self.src_shape = src_shape
        self.dst_shape = dst_shape

    def adjust_args(self, mat: NDArray[np.floating], src_shape: Shape):
        # Calculate bounding box
        w, h = src_shape
        corner_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        corner_dst = cv2.perspectiveTransform(
            corner_src.reshape(-1, 1, 2), mat
        ).reshape(-1, 2)
        coord_min = corner_dst.min(axis=0)
        coord_max = corner_dst.max(axis=0)
        # Shift the transform to make sure all coordinates are positive
        adjust_mat = np.eye(3)
        adjust_mat[0:2, 2] -= coord_min
        # Adjust the shape to make sure all corner inside
        dst_shape = tuple((coord_max - coord_min).astype(int))
        return adjust_mat, dst_shape

    def adjust_shape(
        self,
        src_shape: Shape | None = None,
        dst_shape: Shape | None = None,
    ):
        """Adjust the transform and dst_shape to make sure all corners are inside"""
        if src_shape is not None or dst_shape is not None:
            self.src_shape = src_shape
            self.dst_shape = dst_shape
        if self.dst_shape is None and self.src_shape is not None:
            adjust_mat, shape = self.adjust_args(self.mat, self.src_shape)
            # Apply adjust matrix
            self.follow(adjust_mat)
            self.dst_shape = shape
        elif self.src_shape is None and self.dst_shape is not None:
            adjust_mat, shape = self.adjust_args(
                np.linalg.inv(self.mat), self.dst_shape
            )
            # Apply adjust matrix
            self.precede(adjust_mat)
            self.src_shape = shape

    def __matmul__(self, transform: "np.ndarray | PixelTransform"):
        if isinstance(transform, np.ndarray):
            return PixelTransform(self.mat @ transform, None, self.dst_shape)
        return PixelTransform(
            self.mat @ transform.mat, transform.src_shape, self.dst_shape
        )

    def __rmatmul__(self, transform: "np.ndarray | PixelTransform"):
        if isinstance(transform, np.ndarray):
            return PixelTransform(transform @ self.mat, self.src_shape, None)
        return PixelTransform(
            transform.mat @ self.mat, self.src_shape, transform.dst_shape
        )

    def follow(self, transform: "np.ndarray | PixelTransform"):
        """inplace followed by a transform"""
        self.dst_shape = None
        if isinstance(transform, PixelTransform):
            self.dst_shape = transform.dst_shape
            transform = transform.mat
        self.mat = transform @ self.mat

    def precede(self, transform: "np.ndarray | PixelTransform"):
        """inplace precede a transform beforehand"""
        self.src_shape = None
        if isinstance(transform, PixelTransform):
            self.src_shape = transform.src_shape
            transform = transform.mat
        self.mat = self.mat @ transform

    def warp(self, image: np.ndarray):
        """
        Warp an image using this transform.
        The output shape will be the same as the source image unless a destination shape was specified.
        """
        src_shape = image.shape[-2::-1]
        if self.src_shape is not None:
            assert src_shape == self.src_shape
        dst_shape = src_shape if self.dst_shape is None else self.dst_shape
        return cv2.warpPerspective(image, self.mat, dst_shape)

    def warp_inverse(self, image: np.ndarray):
        """
        Warp an image using the inverse of this transform.
        The output shape will be the same as the destination image unless a source shape was specified.
        """
        dst_shape = image.shape[-2::-1]
        if self.dst_shape is not None:
            assert dst_shape == self.dst_shape
        src_shape = dst_shape if self.src_shape is None else self.src_shape
        return cv2.warpPerspective(image, np.linalg.inv(self.mat), src_shape)

    def apply(self, coords: np.ndarray):
        """Apply the transform to a set of coordinates."""
        if coords.ndim == 1:
            coords = coords[None, :]
        return cv2.perspectiveTransform(coords.reshape(-1, 1, 2), self.mat).reshape(
            -1, 2
        )

    def apply_inverse(self, coords: np.ndarray):
        """Apply the inverse of the transform to a set of coordinates."""
        if coords.ndim == 1:
            coords = coords[None, :]
        return cv2.perspectiveTransform(
            coords.reshape(-1, 1, 2), np.linalg.inv(self.mat)
        ).reshape(-1, 2)


class CRSTransform:
    """Pixel to CRS transform"""

    # 3x3 matrix xy->crs NOT ij->crs
    mat: NDArray[np.floating]
    crs: str = "EPSG:4326"

    def __init__(
        self,
        crs_trans: "tuple[float, float] | NDArray[np.floating] | CRSTransform",
        crs: str = "EPSG:4326",
    ):
        self.mat = np.zeros((3, 3))
        self.crs = crs

        if isinstance(crs_trans, CRSTransform):
            self.mat[:] = crs_trans.mat
            self.crs = crs_trans.crs
        elif isinstance(crs_trans, np.ndarray):
            if crs_trans.shape == (3, 3):
                self.mat = crs_trans
            elif crs_trans.size == 2:
                self.mat[0:2, 2] = crs_trans.ravel()
            elif crs_trans.shape == (2, 3):
                self.mat[0:2] = crs_trans
            else:
                raise ValueError(f"Unsupport array shape: {crs_trans.shape}")
        else:
            self.mat[0:2, 2] = crs_trans

    @property
    def resolution(self) -> float:
        """Pixel resolution (meters per pixel)"""
        corner = np.array([[0, 0], [1, 1]],np.float64)

        diagonal = ((corner[0] - corner[1]) ** 2).sum()  # diagonal pixels

        corner_lonlat = cv2.perspectiveTransform(
            corner.reshape(-1, 1, 2), self.mat
        ).reshape(-1, 2)
        lons, lats = corner_lonlat[:, 0], corner_lonlat[:, 1]
        # 统一转换到WGS84
        if self.crs != "EPSG:4326":
            # 创建转换器（转WGS84）
            to_wgs84 = Transformer.from_crs(
                crs_from=self.crs,
                crs_to="EPSG:4326",  # WGS84
                always_xy=True,  # 强制使用(x=经度, y=纬度)
            )
            lons, lats = to_wgs84.transform(lons, lats)
        # 计算距离
        geod = Geod(ellps="WGS84")  # 使用WGS84椭球体
        lon1, lon2 = lons
        lat1, lat2 = lats
        _, _, distance = geod.inv(lon1, lat1, lon2, lat2)

        return distance / diagonal

    def precede(self, transform: np.ndarray):
        """inplace precede a transform beforehand"""
        self.mat @= transform


class Transform(PixelTransform):
    """Consecutive pixel transform and crs transform"""

    crs: CRSTransform

    def __init__(
        self,
        crs: CRSTransform,
        pix_mat: NDArray[np.floating] | None = None,
        src_shape: Shape | None = None,
        dst_shape: Shape | None = None,
    ):
        super().__init__(pix_mat, src_shape, dst_shape)
        if not isinstance(crs, CRSTransform):
            crs = CRSTransform(crs)
        self.crs = crs

    def follow(self, transform: "np.ndarray | PixelTransform"):
        """inplace followed by a pixel transform"""
        super().follow(transform)
        if isinstance(transform, PixelTransform):
            transform = transform.mat
        self.crs.precede(np.linalg.inv(transform))

    @property
    def combined(self):
        return CRSTransform(self.crs.mat @ self.mat, self.crs.crs)
