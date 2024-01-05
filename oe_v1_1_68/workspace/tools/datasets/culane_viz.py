"""viz culane."""

import argparse
import pickle

from hat.data.datasets.culane_dataset import CuLaneDataset
from hat.utils.logger import init_logger
from hat.visualize.lane_lines import LanelineViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=50,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".hat_logs/laneline_viz")

    dataset = CuLaneDataset(
        data_path=args.data_path,
        to_rgb=True,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = LanelineViz(is_plot=args.plot)
    for i, data in enumerate(dataset):

        img = data["ori_img"]
        gt_lines = data["ori_gt_lines"]
        print(data["image_name"])
        if args.plot:
            viz(img, gt_lines)
        if i > int(args.viz_num):
            break
