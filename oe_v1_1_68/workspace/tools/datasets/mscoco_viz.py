"""viz mscoco."""

import argparse
import pickle

import torch

from hat.data.datasets.mscoco import Coco
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
        default=5000,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".hat_logs/coco_viz")

    dataset = Coco(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = DetViz(is_plot=args.plot)

    for i, data in enumerate(dataset):
        img = data["img"]
        gt_bboxes = torch.from_numpy(data["gt_bboxes"])
        gt_classes = torch.from_numpy(data["gt_classes"])
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
