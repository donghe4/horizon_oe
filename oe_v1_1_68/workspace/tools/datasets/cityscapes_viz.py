"""viz cityscapes."""

import argparse
import pickle

import torchvision

from hat.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS, Cityscapes
from hat.data.transforms.common import PILToTensor
from hat.data.transforms.segmentation import LabelRemap
from hat.utils.logger import init_logger
from hat.visualize.seg import SegViz

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

    init_logger(".hat_logs/cityscapes_viz")

    dataset = Cityscapes(
        data_path=args.data_path,
        transforms=torchvision.transforms.Compose(
            [
                PILToTensor(),
                LabelRemap(mapping=CITYSCAPES_LABLE_MAPPINGS),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = SegViz(is_plot=args.plot)

    for i, data in enumerate(dataset):
        img = data["img"]
        gt_seg = data["gt_seg"]
        if args.plot:
            viz(img, gt_seg)
        if i > int(args.viz_num):
            break
