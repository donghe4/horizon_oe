import argparse
import os

import numpy as np
from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        default=None,
        help="sensor to ego translation",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=".",
        help="path to save generate npy",
    )
    parser.add_argument(
        "--save-by-city",
        action="store_true",
    )
    parser.add_argument(
        "--version", default="v1.0-mini", help="The version to packed."
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


cam_names = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def gen(version, data_path, save_path, save_by_city=True):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    scenes = nusc.scene
    dirs = []
    for sc in scenes:
        sensor2ego_translation = []
        sensor2ego_rotation = []
        camera_intrinsic = []
        name = sc["name"]
        sample = nusc.get("sample", sc["first_sample_token"])
        data = sample["data"]
        loc = nusc.get("log", sc["log_token"])["location"]
        city = loc.split("-")[0]
        for cam in cam_names:
            sd_rec = nusc.get("sample_data", data[cam])
            cs_rec = nusc.get(
                "calibrated_sensor", sd_rec["calibrated_sensor_token"]
            )
            sensor2ego_translation.append(cs_rec["translation"])
            sensor2ego_rotation.append(cs_rec["rotation"])
            camera_intrinsic.append(cs_rec["camera_intrinsic"])
        sensor2ego_translation_array = np.stack(sensor2ego_translation)
        sensor2ego_rotation_array = np.stack(sensor2ego_rotation)
        camera_intrinsic_array = np.stack(camera_intrinsic)

        if save_by_city is True:
            name = city

        sc_dir = os.path.join(save_path, name)
        os.makedirs(sc_dir, exist_ok=True)

        dirs.append(sc_dir)
        print(f"Save Scene {name} ...")
        sensor2ego_translation_path = os.path.join(
            sc_dir, "sensor2ego_translation.npy"
        )
        np.save(sensor2ego_translation_path, sensor2ego_translation_array)

        sensor2ego_rotation_array_path = os.path.join(
            sc_dir, "sensor2ego_rotation.npy"
        )
        np.save(sensor2ego_rotation_array_path, sensor2ego_rotation_array)

        camera_intrinsic_path = os.path.join(sc_dir, "camera_intrinsic.npy")
        np.save(camera_intrinsic_path, camera_intrinsic_array)
    return list(set(dirs))


if __name__ == "__main__":
    args, args_env = parse_args()
    gen(args.version, args.data_path, args.save_path, args.save_by_city)
