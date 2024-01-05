import argparse
import logging
import os
from collections import Sequence

import horizon_plugin_pytorch as horizon
import numpy as np
import torch

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--homography",
        "-o",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path for generating npy.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo


def init_model(config):
    cfg = Config.fromfile(config)

    horizon.march.set_march(cfg.march)
    with RegistryContext():
        model = cfg.deploy_model
        model["compile_model"] = False
        model = build_from_registry(model)
        model.eval()

        predictor = cfg.align_bpu_predictor
        pipeline = predictor["model_convert_pipeline"]
        pipeline = build_from_registry(pipeline)
        model = pipeline(model).cuda()
    return model, cfg


def gen(model, cfg, homography, save_path):
    logger.info("=" * 50 + "Generating reference points..." + "=" * 50)
    resize_size = cfg.resize_shape[1:]
    input_size = cfg.val_data_shape[1:]
    orig_size = cfg.orig_shape[1:]

    vt_input_hw = cfg.vt_input_hw
    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)

    scale = (resize_size[0] / orig_size[0], resize_size[1] / orig_size[1])

    homo = np.load(homography)
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))

    deploy_inputs = cfg.deploy_inputs
    inputs = {"img": deploy_inputs["img"].cuda()}
    inputs["ego2img"] = torch.tensor(homo).cuda()

    ref_p = model.export_reference_points(inputs, vt_input_hw)

    for k, p in ref_p.items():
        path = os.path.join(save_path, f"{k}.npy")
        print(f"Saving {path}...")
        np.save(path, p)

    logger.info("=" * 50 + "END ONNX" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    model, cfg = init_model(args.config)
    gen(model, cfg, args.homography, args.save_path)
