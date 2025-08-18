"""
Interactive Demonstration Using UAV-VisLoc Dataset

Control keys
-   Esc: Exit
-   A: Previous image
-   W: 10th Previous image
-   D: Next image
-   S: 10th Next image
"""

from dataclasses import asdict
from pathlib import Path

from pyproj import Transformer
import cv2
import numpy as np

from uavcalibration.calibration import Calibration
from uavcalibration.datasets import *
from uavcalibration.map import *
from uavcalibration.transform import *

project_root = Path(__file__).parent.parent
dataset = VisLocDataset(project_root / "datasets" / "UAV_VisLoc_example")
satellite_map: Map
satellite_map = GeoTiffMap([i.image_path for i in dataset.satellite_infos.values()])
# satellite_map = TiledMap(r"http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}")

# 全局变量
uav_data: UAVData
satellite_info = np.zeros((1, 1, 3), np.uint8), CRSTransform((0, 0))
# 数据位置
data_index = 0
# 存储鼠标位置
uav_pos = (0, 0)
satellite_pos = (0, 0)
lon = 0
lat = 0


def calibrate():
    global satellite_info
    uav_image = uav_data.uav_image

    calibration = Calibration(uav_image)
    calibration.coarse_calibrate(**asdict(uav_data))
    src_shape = uav_image.shape
    calibration.transform.adjust_shape(src_shape=(src_shape[1], src_shape[0]))

    tmp_transform = calibration.transform.combined
    h, w = calibration.uav_image.shape[:2]
    with satellite_map:
        satellite_info = satellite_map.get(
            tmp_transform.bounds(h=h, w=w),
            tmp_transform.crs,
            resolution=tmp_transform.resolution,
        )

    calibration.fine_calibrate(*satellite_info)
    return calibration


def update_lonlat():
    global uav_pos, satellite_pos, lon, lat
    trans = calibration.transform
    lon, lat = tuple(trans.crs.apply(np.array(satellite_pos, np.float64)).ravel())
    transformer = Transformer.from_crs(trans.crs.crs, "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(lon, lat)


# 鼠标回调函数
def uav_callback(event, x, y, flags, param):
    global uav_pos, satellite_pos, lon, lat
    if event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动事件
        uav_pos = (x, y)
        satellite_pos = tuple(
            calibration.transform.apply(np.array(uav_pos, np.float64))
            .ravel()
            .astype(int)
        )
        update_lonlat()


def satellite_callback(event, x, y, flags, param):
    global uav_pos, satellite_pos, lon, lat
    if event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动事件
        satellite_pos = (x, y)
        uav_pos = tuple(
            calibration.transform.apply_inverse(np.array(satellite_pos, np.float64))
            .ravel()
            .astype(int)
        )
        update_lonlat()


def put_text(image: np.ndarray, text: str, size: int, color):
    text_org = (size * 5, size * 15)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size * 0.5
    thickness = int(size)
    cv2.putText(image, text, text_org, font_face, font_scale, color, thickness)


def draw_cross(image: np.ndarray, pos: tuple[int, int], size: int, color):
    length = size * 5
    # 绘制水平线
    cv2.line(image, (pos[0] - length, pos[1]), (pos[0] + length, pos[1]), color, size)
    # 绘制垂直线
    cv2.line(image, (pos[0], pos[1] - length), (pos[0], pos[1] + length), color, size)
    cv2.circle(image, pos, size, color, size)


uav_data = dataset[data_index]
calibration = calibrate()

# 创建窗口并绑定回调函数
cv2.namedWindow("UAV Image", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Satellite Image", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("UAV Image", uav_callback)
cv2.setMouseCallback("Satellite Image", satellite_callback)

while True:
    # 在图像上绘制坐标
    uav_image = uav_data.uav_image[..., ::-1].copy()
    satellite_image = satellite_info[0][..., ::-1].copy()

    size = max(uav_image.shape[::2]) // 300
    color = (0, 255, 0)
    put_text(uav_image, f"Position: {uav_pos}", size, color)
    put_text(satellite_image, f"lon: {lon:10.6f} lat: {lat:10.6f}", size, color)
    draw_cross(satellite_image, satellite_pos, size, color)
    draw_cross(uav_image, uav_pos, size, color)
    # 显示图像
    cv2.imshow("UAV Image", uav_image)
    cv2.imshow("Satellite Image", satellite_image)
    key = cv2.waitKey(1)
    # 按ESC键退出
    if key == 27:
        break
    elif key in [119, 97, 115, 100]:
        if key in [
            119,
        ]:  # w
            data_index -= 10
        elif key in [97]:  # a
            data_index -= 1
        elif key in [115]:  # s
            data_index += 10
        elif key in [100]:  # d
            data_index += 1
        put_text(uav_image, f"Processing ...", size * 3, color)
        cv2.imshow("UAV Image", uav_image)
        cv2.waitKey(1)
        uav_data = dataset[data_index]
        calibration = calibrate()
cv2.destroyAllWindows()
