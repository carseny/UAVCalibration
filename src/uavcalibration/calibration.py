import numpy as np

from .types import *
from .graph import *
from .transform import *
from . import rectification as rect
from .matching import *


def match_transform(
    image_src: ImageMat,
    image_dst: ImageMat,
    resolution: float,  # meters per pixel
    tolerance=5.0,  # meters
    *args,
    **kwargs,
) -> np.ndarray:
    match_result = match_images(image_src, image_dst, *args, **kwargs)
    threshold = tolerance / resolution  # tolerance threshold
    homography_result = match_homography(
        match_result.kpts0, match_result.kpts1, threshold
    )
    return homography_result.mat


class Calibration:
    uav_image: ImageMat
    transform: Transform

    def __init__(
        self,
        # uav image args
        uav_image: ImageMat,
    ):
        self.uav_image = uav_image

    @property
    def calibrated_image(self):
        return self.transform.warp(self.uav_image)

    def coarse_calibrate(
        self,
        # uav crs args
        longitude: float,  # in degrees
        latitude: float,  # in degrees
        # camera args
        focal_length: float | None = None,  # in pixels
        camera_mat: np.ndarray | None = None,  # xyz -> x'y'1
        # angle args
        yaw: float | None = None,  # in radians
        pitch: float | None = None,  # in radians
        roll: float | None = None,  # in radians
        rotate_mat: np.ndarray | None = None,  # xyz -> x'y'z'
        # resolution args
        height: float = 100,  # in meters
        *args,
        **kwargs,
    ):
        raw_shape: Shape = self.uav_image.shape[1], self.uav_image.shape[0]

        if camera_mat is None:
            camera_mat = rect.camera_mat(raw_shape, focal_length)
        if rotate_mat is None:
            rotate_mat = rect.rotate_mat(yaw=yaw, pitch=pitch, roll=roll)
        rect_mat = rect.rectify_mat(camera_mat=camera_mat, rotate_mat=rotate_mat)
        crs_trans = rect.crs_trans(
            longitude=longitude,
            latitude=latitude,
            camera_mat=camera_mat,
            rotate_mat=rotate_mat,
            height=height,
        )
        self.transform = Transform(
            pix_mat=rect_mat,
            src_shape=raw_shape,
            crs=crs_trans,
        )
        self.transform.adjust_shape()

    def fine_calibrate(
        self,
        # satellite args
        satellite_image: ImageMat,
        satellite_crs: CRSTransform,
        # match args
        *args,
        **kwargs,
    ):
        mat = match_transform(
            self.calibrated_image,
            satellite_image,
            resolution=self.transform.crs.resolution,
            *args,
            **kwargs,
        )
        self.transform.follow(mat)
        self.transform.dst_shape = satellite_image.shape[1], satellite_image.shape[0]
        self.transform.crs = satellite_crs
