import numpy as np


def rectification_mat(
    w: int,
    h: int,
    pitch: float | None = None,
    roll: float | None = None,
    yaw: float | None = None,
    focal_length: float | None = None,
):
    # TODO: height
    
    # Camera intrinsic matrix
    cam_mat = np.array(
        [
            [focal_length, 0, w // 2],
            [0, focal_length, h // 2],
            [0, 0, 1],
        ],
    )

    # Rotation matrices
    rot_mat = np.eye(3)
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

    # Final rectification matrix
    return cam_mat @ rot_mat @ np.linalg.inv(cam_mat)
