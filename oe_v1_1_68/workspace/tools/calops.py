"""Calculate ops of torch model."""
import argparse
import copy

import horizon_plugin_pytorch as horizon
import torch

from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.setup_env import setup_args_env
from hat.utils.statistics import cal_ops


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input-shape",
        default=None,
        type=str,
        help="input size in B,H,W,C order",
    )
    parser.add_argument(
        "--method",
        default="fx",
        type=str,
        help="method used by calops, either hook or fx",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def calops(
    config: str,
    input_shape: str = None,
    args_env: list = None,
    method: str = None,
):
    """Calculate Flops and Params function.

    Args:
        config: Config file path.
        input_shape: Input size in B,H,W,C order.
        args_env: The args will be set in env.
        method: The method used by calops, can be `fx` or `hook`.
    """
    if args_env:
        setup_args_env(args_env)
    config = Config.fromfile(config)
    if hasattr(config, "deploy_model") and config.deploy_model:
        model_cfg = copy.deepcopy(config.deploy_model)
    else:
        model_cfg = copy.deepcopy(config.model)

    if hasattr(config, "calops_cfg"):
        calops_cfg = config.calops_cfg
        if "method" in calops_cfg:
            method = calops_cfg["method"]
        elif "input_shape" in calops_cfg:
            input_shape = calops_cfg["input_shape"]

    model = build_from_registry(model_cfg)
    model.eval()

    if input_shape is not None:
        assert "," in input_shape
        input_shape = [int(i) for i in input_shape.split(",")]
    else:
        if "deploy_inputs" in config:
            pass
        else:
            raise ValueError("Please provide input shape by --input-shape")

    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    if "deploy_inputs" in config:
        fake_data = config.deploy_inputs
    else:
        fake_data = torch.randn(input_shape)
    total_ops, total_params = cal_ops(model, fake_data, method)
    return total_ops, total_params


if __name__ == "__main__":
    args, args_env = parse_args()
    total_ops, total_params = calops(
        config=args.config,
        input_shape=args.input_shape,
        args_env=args_env,
        method=args.method,
    )
    print("Params: %.6f M" % (total_params / (1000 ** 2)))
    print("FLOPs: %.6f G" % (total_ops / (1000 ** 3)))
