"""viz sceneflow."""

import argparse
import pickle

from hat.data.datasets.sceneflow_dataset import SceneFlow
from hat.utils.logger import init_logger
from hat.visualize.disparity import DispViz

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

    init_logger(".hat_logs/sceneflow_viz")

    dataset = SceneFlow(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = DispViz(is_plot=args.plot)

    for i, data in enumerate(dataset):

        viz(data["img"], data["gt_disp"])

        if i > int(args.viz_num):
            break
