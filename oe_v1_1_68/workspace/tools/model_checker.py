# Copyright (c) Horizon Robotics. All rights reserved.

import argparse
import logging
import os
import warnings

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.quantization import check_model

from hat.registry import build_from_registry
from hat.utils.config import Config, ConfigVersion
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

logger = logging.getLogger(__file__)


def model_checker(cfg_file, args_env=None):
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(cfg_file)

    # check config version
    config_version = cfg.get("VERSION", None)
    if config_version is not None:
        assert (
            config_version == ConfigVersion.v2
        ), "{} only support config with version 2, not version {}".format(
            os.path.basename(__file__), config_version.value
        )
    else:
        warnings.warn(
            "VERSION will must set in config in the future."
            "You can refer to configs/classification/resnet18.py,"
            "and configs/classification/bernoulli/mobilenetv1.py."
        )

    if not hasattr(cfg, "march"):
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))
    deploy_model = cfg.deploy_model
    deploy_inputs = cfg.deploy_inputs
    if hasattr(cfg, "deploy_model_convert_pipeline"):
        deploy_model_convert_pipeline = cfg.deploy_model_convert_pipeline
    else:
        deploy_model_convert_pipeline = dict(  # noqa: C408
            type="ModelConvertPipeline",
            qat_mode="fuse_bn",
            converters=[
                dict(type="Float2QAT"),  # noqa: C408
                dict(type="QAT2Quantize"),  # noqa: C408
            ],
        )
    deploy_model = build_from_registry(deploy_model)
    model_convert_pipeline = build_from_registry(deploy_model_convert_pipeline)
    deploy_model = model_convert_pipeline(deploy_model)
    flag = check_model(deploy_model, deploy_inputs, advice=10)
    if flag != 0:
        raise AssertionError("Failed to pass hbdk checker")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    model_checker(args.config, args_env=args_env)
