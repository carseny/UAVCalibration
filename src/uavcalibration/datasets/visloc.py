from .dataset import UAVData, SatelliteData, UAVDataset

import numpy as np
import cv2

import csv
from pathlib import Path


class VisLocDataset(UAVDataset):
    def get_satellite_image(self, partition: str):
        return self.satellite_datas[partition].image

    def get_satellite_area(self, uav_data: UAVData):
        image = self.get_satellite_image(uav_data.partition)
        center_w, center_h = self.satellite_datas[uav_data.partition].lonlat2wh(
            uav_data.lon, uav_data.lat
        )
        scale = uav_data.height / 1240
        h = uav_data.img_h * scale
        w = uav_data.img_w * scale / np.cos(np.radians(uav_data.lat))
        return image[
            int(center_h - h // 2) : int(center_h + h // 2),
            int(center_w - w // 2) : int(center_w + w // 2),
        ]

    def read_folder(self, folder: Path):
        partition = folder.name
        with (folder / (partition + ".csv")).open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.uav_datas.append(
                    UAVData(folder.parent, partition, **UAVData.convert_dict(row))
                )

    def __init__(self, dataset_path: str | Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        self.uav_datas: list[UAVData] = []
        self.satellite_datas: dict[str, SatelliteData] = {}
        with (dataset_path / "satellite_coordinates_range.csv").open("r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                partition = row["mapname"].rsplit(".")[0]
                self.satellite_datas[partition] = SatelliteData(
                    dataset_path, partition, **SatelliteData.convert_dict(row)
                )
        for folder in dataset_path.iterdir():
            if folder.is_dir():
                self.read_folder(folder)

    def __getitem__(self, index) -> UAVData:
        return self.uav_datas[index]

    def __len__(self) -> int:
        return len(self.uav_datas)

    def __iter__(self):
        return iter(self.uav_datas)


if __name__ == "__main__":
    dataset = VisLocDataset("datasets/UAV_VisLoc_example/")
    print(max(dataset, key=lambda x: x.Phi2))
    print(min(dataset, key=lambda x: x.Phi2))
    for d in dataset:
        print(
            f"lat: {d.lat:.5f}, lon: {d.lon:.5f}, yaw: {d.yaw:.2f}, pitch: {d.pitch:6.3f}, roll: {d.roll:6.3f}"
        )
        image = d.image
        corrected = cv2.warpPerspective(image, d.perspect_mat, (d.img_w, d.img_h))
        satellite = dataset.get_satellite_area(d)
        h, w = image.shape[:2]
        view_shape = (w // 5, h // 5)
        cv2.imshow("Original", cv2.resize(image, view_shape)[..., ::-1])
        cv2.imshow("Corrected", cv2.resize(corrected, view_shape)[..., ::-1])
        cv2.imshow("Satellite", cv2.resize(satellite, view_shape)[..., ::-1])
        cv2.waitKey(0)
