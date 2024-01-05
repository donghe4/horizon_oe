"""Build kitti3d."""

import argparse
import os

from hat.data.datasets.kitti3d import Kitti3DDetectionPacker
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Pack kitti3d dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The target pack type for result of packer",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="train", help="The split to pack."
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/kitti3d_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir
    split_name = args.split_name

    pack_path = os.path.join(
        args.target_data_dir,
        "%s_%s" % (split_name, args.pack_type),
    )

    packer = Kitti3DDetectionPacker(
        directory,
        pack_path,
        args.split_name,
        int(args.num_workers),
        args.pack_type,
        None,
    )
    packer()
