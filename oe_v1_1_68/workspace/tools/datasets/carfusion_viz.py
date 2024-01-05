"""viz imagenet."""

import argparse
import pickle

import numpy as np

from hat.data.datasets.carfusion_keypoints_dataset import CarfusionPackData
from hat.utils.logger import init_logger
from hat.visualize.keypoints import KeypointsViz

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

    init_logger(".hat_logs/carfusion_viz")

    dataset = CarfusionPackData(
        data_path=args.data_path,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = KeypointsViz(
        is_plot=args.plot,
    )

    for i, data in enumerate(dataset):
        img = data["img"]
        gt_ldmk = data["gt_ldmk"]
        placehold = np.ones([12, 3])
        placehold[:, :2] = gt_ldmk

        print(img.shape, gt_ldmk.shape)
        viz(img, placehold, save_path=f".hat_logs/carfusion_vis_imgs/{i}.png")
        if i + 1 >= int(args.viz_num):
            break
