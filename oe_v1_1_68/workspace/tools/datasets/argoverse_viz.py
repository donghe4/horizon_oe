"""viz nuscenes."""

import argparse
import logging
import pickle

from hat.data.datasets.argoverse_dataset import Argoverse1Dataset
from hat.utils.logger import init_logger
from hat.visualize import ArgoverseViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--map-path",
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

    init_logger(".hat_logs/argoverse_viz", level=logging.WARNING)
    dataset = Argoverse1Dataset(
        data_path=args.data_path, map_path=args.map_path
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = ArgoverseViz()

    for i, data in enumerate(dataset):
        traj_feat_mask = data["feat_mask"]
        traj_feat = data["traj_feat"].transpose((2, 1, 0))
        traj_mask = data["traj_mask"]
        labels = data["traj_labels"]

        lane_feat = data["lane_feat"].transpose((2, 1, 0))
        lane_mask = data["lane_mask"]

        viz(traj_feat_mask, traj_feat, traj_mask, lane_feat, lane_mask, labels)
        if i > int(args.viz_num):
            break
