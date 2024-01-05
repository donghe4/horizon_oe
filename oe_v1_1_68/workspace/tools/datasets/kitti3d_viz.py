"""viz kitti3d."""

import argparse
import pickle

import torch

from hat.data.datasets.kitti3d import Kitti3D
from hat.utils.logger import init_logger
from hat.visualize.lidar_det import lidar_det_visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=1,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".hat_logs/kitti3d_viz")

    dataset = Kitti3D(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))

    for i, data in enumerate(dataset):
        points = data["lidar"]["points"]
        # bbox_2d = data["metadata"]["bbox"]
        bbox3d_on_lidar = torch.from_numpy(
            data["lidar"]["annotations"]["boxes"]
        )
        gt_labels = torch.from_numpy(data["metadata"]["category_id"])
        gt_score = torch.ones(bbox3d_on_lidar.shape[0])
        gt_data = (
            bbox3d_on_lidar,
            gt_labels,
            gt_score,
        )

        lidar_det_visualize(points, gt_data, is_plot=args.plot)
        if i > int(args.viz_num):
            break
