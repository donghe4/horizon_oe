import argparse
import os

import numpy as np
from pyquaternion import Quaternion


def get_homography_by_cam(
    sensor2ego_translation, sensor2ego_rotation, camera_intrinsic
):
    rotation = Quaternion(sensor2ego_rotation).rotation_matrix
    ego2sensor_r = np.linalg.inv(rotation)
    ego2sensor_t = sensor2ego_translation @ ego2sensor_r.T
    ego2sensor = np.eye(4)
    ego2sensor[:3, :3] = ego2sensor_r.T
    ego2sensor[3, :3] = -np.array(ego2sensor_t)

    camera_intrinsic = np.array(camera_intrinsic)

    viewpad = np.eye(4)
    viewpad[
        : camera_intrinsic.shape[0], : camera_intrinsic.shape[1]
    ] = camera_intrinsic
    ego2img = viewpad @ ego2sensor.T
    return ego2img


def _gen(s2e_t, s2e_r, cam_intrin):
    homography = []
    for each_s2e_t, each_s2e_r, each_cam_int in zip(s2e_t, s2e_r, cam_intrin):
        homography.append(
            get_homography_by_cam(each_s2e_t, each_s2e_r, each_cam_int)
        )
    homography = np.array(homography)

    return homography


def gen(s2e_t_path, s2e_r_path, cam_intrin_path, save_path):
    s2e_t = np.load(s2e_t_path)
    s2e_r = np.load(s2e_r_path)
    cam_intrin = np.load(cam_intrin_path)
    homo = _gen(s2e_t, s2e_r, cam_intrin)
    homo_path = os.path.join(save_path, "homography.npy")
    np.save(homo_path, homo)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor2ego-translation",
        type=str,
        required=True,
        default=None,
        help="sensor to ego translation",
    )
    parser.add_argument(
        "--sensor2ego-rotation",
        type=str,
        required=True,
        default=None,
        help="sensor to ego rotation",
    )
    parser.add_argument(
        "--camera-intrinsic",
        type=str,
        required=True,
        default=None,
        help="comera intrinsic",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=".",
        help="Save path for generate npy.",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    gen(
        args.sensor2ego_translation,
        args.sensor2ego_rotation,
        args.camera_intrinsic,
        args.save_path,
    )
