import argparse

from hat.data.datasets.kitti3d import (
    create_groundtruth_database,
    create_kitti_info_file,
    create_reduced_point_cloud,
)
from hat.data.datasets.nuscenes_dataset import (
    create_nuscenes_groundtruth_database,
    create_nuscenes_infos,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--extra-tag", type=str, default="nuscenes")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./tmp_data/nuscenes",
        required=False,
        help="path of output dbinfo pkl",
    )

    return parser.parse_args()


def kitti_data_preprocess(root_path):
    create_kitti_info_file(root_path)
    create_reduced_point_cloud(root_path)
    create_groundtruth_database(root_path)


def nuscenes_data_prep(dataset_name, dataset_path, info_prefix, out_dir):
    """Prepare database file related to nuScenes dataset."""
    create_nuscenes_groundtruth_database(
        dataset_name, dataset_path, info_prefix, out_dir
    )
    create_nuscenes_infos(dataset_name, dataset_path, info_prefix, out_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "kitti3d":
        kitti_data_preprocess(args.root_dir)
    elif args.dataset == "nuscenes":
        nuscenes_data_prep(
            dataset_name="NuscenesLidarDataset",
            dataset_path=args.root_dir,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
        )
