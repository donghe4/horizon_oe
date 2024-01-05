"""viz nuscenes."""

import argparse
import logging
import pickle

import numpy as np
import torch
from torchvision.transforms import functional as F

from hat.core.nus_box3d_utils import bbox_bev2ego
from hat.data.datasets.nuscenes_dataset import NuscenesBevDataset
from hat.utils.logger import init_logger
from hat.visualize.nuscenes import NuscenesViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--meta-path",
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
    parser.add_argument("--bev-size", default=[51.2, 51.2, 0.8], type=list)
    parser.add_argument("--map-size", default=[15, 30, 0.15], type=list)

    args = parser.parse_args()

    init_logger(".hat_logs/nuscenes_viz", level=logging.WARNING)
    dataset = NuscenesBevDataset(
        data_path=args.data_path,
        bev_size=args.bev_size,
        map_size=args.map_size,
        map_path=args.meta_path,
        with_bev_bboxes=True,
        with_bev_mask=True,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = NuscenesViz(bev_size=args.bev_size, is_plot=args.plot)

    for i, data in enumerate(dataset):
        imgs = []
        for img, name in zip(data["img"], data["cam_name"]):
            img = F.pil_to_tensor(img)
            imgs.append({"name": name, "img": img})

        mask = torch.tensor(data["bev_seg_indices"]).unsqueeze(0)
        preds = {"bev_seg": mask}

        meta = {"ego2img": data["ego2img"]}

        bbox = data["bev_bboxes_labels"]
        bbox = np.concatenate(
            [bbox[:, :9], np.ones((bbox.shape[0], 1)), bbox[:, 9:]], axis=1
        )

        bbox = bbox_bev2ego(bbox, args.bev_size)

        preds["bev_det"] = [bbox]

        viz(imgs, preds, meta)
        if i > int(args.viz_num):
            break
