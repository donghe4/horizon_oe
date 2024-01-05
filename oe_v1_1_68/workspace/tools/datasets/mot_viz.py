"""viz mot17."""

import argparse

import torch

from hat.data.datasets.mot17_dataset import Mot17Dataset
from hat.utils.logger import init_logger
from hat.visualize.det import DetViz

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

    init_logger(".hat_logs/mot17_viz")
    dataset = Mot17Dataset(
        data_path=args.data_path,
    )
    viz = DetViz(is_plot=args.plot)

    for i, data in enumerate(dataset):
        frame_data = data["frame_data_list"][0]
        img = frame_data["img"]
        gt_bboxes = torch.from_numpy(frame_data["gt_bboxes"])
        gt_classes = torch.from_numpy(frame_data["gt_classes"])
        gt_score = torch.ones_like(gt_classes)
        gt_labels = torch.cat(
            (
                gt_bboxes,
                gt_score.unsqueeze(-1),
                gt_classes.unsqueeze(-1),
            ),
            -1,
        )
        viz(img, gt_labels)
        if i > int(args.viz_num):
            break
