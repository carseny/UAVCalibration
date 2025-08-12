import numpy as np
from pyproj import Transformer

from .types import *
from .transform import *

__all__ = ["camera_mat", "rotate_mat", "rectify_mat", "crs_trans"]


def camera_mat(
    shape: Shape,
    focal_length: float | None = None,
):
    w, h = shape
    # If focal length is not provided, set it to a reasonable default value
    if focal_length == None:
        focal_length = max(w, h) * 1.5
    # Camera intrinsic matrix
    cam_mat = np.array(
        [
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return cam_mat


def rotate_mat(
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
):
    """由相机坐标系到地面经纬度三维坐标系的旋转矩阵"""
    # Rotation matrices
    rot_mat = np.eye(3, dtype=np.float64)
    # 旋转顺序待定
    if yaw is not None:
        rot_mat @= np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
    if pitch is not None:
        rot_mat @= np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
    if roll is not None:
        rot_mat @= np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
    return rot_mat


def rectify_mat(
    camera_mat: np.ndarray,
    rotate_mat: np.ndarray,
):
    # Final rectification matrix
    return camera_mat @ rotate_mat @ np.linalg.inv(camera_mat)


def crs_trans(
    longitude: float,
    latitude: float,
    camera_mat: np.ndarray,
    rotate_mat: np.ndarray,
    height: float,  # in meters
) -> CRSTransform:
    # find utm zone
    utm_zone = int((longitude + 180) / 6) + 1
    utm_epsg = 32600 + utm_zone  # 北半球
    if latitude < 0:  # 南半球
        utm_epsg += 100
    utm_crs = f"EPSG:{utm_epsg}"
    # transform wgs84 to utm
    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    target_xy = to_utm.transform(longitude, latitude)

    # calculate the ground point in the camera coordinate system
    ground_point = np.array([0, 0, height])
    pixel2world = rotate_mat @ np.linalg.inv(camera_mat)
    ground_predict = ground_point @ np.linalg.inv(pixel2world).T
    # standardize transform to meters per pixel
    pixel2world *= ground_predict[2]
    # eliminate z axis
    pixel2utm = pixel2world / np.array([[1], [1], [height]])

    # projection offset
    shift = np.eye(3)
    shift[:2, 2] = target_xy

    return CRSTransform(shift @ pixel2utm, utm_crs)
