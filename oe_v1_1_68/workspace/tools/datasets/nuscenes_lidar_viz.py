"""viz kitti3d."""

import argparse
import pickle

import torch

from hat.data.datasets.nuscenes_dataset import NuscenesLidarWithSegDataset
from hat.data.transforms.lidar_utils import AssignSegLabel
from hat.utils.logger import init_logger
from hat.visualize.lidar_det import lidar_det_visualize
from hat.visualize.seg import SegViz

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--num-sweeps",
        default=0,
    )
    parser.add_argument(
        "--viz-num",
        default=-1,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".hat_logs/nuscenes_lidar_viz")

    dataset = NuscenesLidarWithSegDataset(
        num_sweeps=args.num_sweeps,
        data_path=args.data_path,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        use_valid_flag=True,
        classes=class_names,
        transforms=AssignSegLabel(
            bev_size=[512, 512],
            num_classes=2,
            class_names=[0, 1],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            voxel_size=[0.2, 0.2],
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    for i, data in enumerate(dataset):
        points = data["lidar"]["points"]

        # get det gt info
        bbox3d_on_lidar = torch.from_numpy(
            data["lidar"]["annotations"]["boxes"]
        )
        bbox3d_on_lidar = bbox3d_on_lidar[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]
        gt_labels = torch.from_numpy(data["lidar"]["annotations"]["labels"])
        gt_score = torch.ones(bbox3d_on_lidar.shape[0])

        gt_data = {
            "bboxes": bbox3d_on_lidar,
            "scores": gt_score,
            "labels": gt_labels,
        }

        lidar_det_visualize(points, gt_data, is_plot=args.plot, reverse=True)

        # get seg gt info
        gt_seg_labels = torch.from_numpy(
            data["lidar"]["annotations"]["gt_seg_labels"]
        )
        gt_seg_labels.masked_fill_(gt_seg_labels == -1, 2)
        SegViz(is_plot=args.plot)(None, gt_seg_labels)

        if i > int(args.viz_num):
            break
