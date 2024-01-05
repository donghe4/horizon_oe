"""predict tools."""
import argparse
import logging
import os
import warnings
from typing import Sequence, Union

import horizon_plugin_pytorch as horizon
import torch

from hat.engine import build_launcher
from hat.registry import RegistryContext, build_from_registry
from hat.utils.checkpoint import load_state_dict
from hat.utils.config import Config, ConfigVersion
from hat.utils.distributed import get_dist_info
from hat.utils.logger import (
    LOG_DIR,
    DisableLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
    rank_zero_info,
)
from hat.utils.seed import seed_training
from hat.utils.setup_env import setup_args_env, setup_hat_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the predict stage, you should define "
            "{stage}_predictor in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="predict config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0,1,2,3', "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        choices=["NCCL", "GLOO"],
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path used to predict",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export HAT_PIPELINE_TEST=1, which used in config",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def predict_entrance(
    device: Union[None, int, Sequence[int]],
    stage: str,
    cfg_file: str,
    ckpt: str = None,
):
    cfg = Config.fromfile(cfg_file)
    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step=stage,
        prefix="predict-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    if cfg.get("seed", None) is not None:
        seed_training(cfg.seed)

    rank_zero_info("=" * 50 + "BEGIN %s PREDICT" % stage.upper() + "=" * 50)
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    # build model
    assert hasattr(
        cfg, f"{stage}_predictor"
    ), f"you should define {stage}_predictor in the config file"
    predictor = cfg[f"{stage}_predictor"]

    with DisableLogger(disable_logger), RegistryContext():
        predictor["device"] = device
        predictor = build_from_registry(predictor)
        if ckpt is not None:
            logger.warning("Make sure ckpt is consistent with training stage")
            model = predictor.model
            load_pred_ckpt_func = cfg.get(
                "load_pred_ckpt_func", load_state_dict
            )
            load_pred_ckpt_func(
                model,
                path_or_dict=ckpt,
                map_location="cpu",
            )
            predictor.model = model
        predictor.fit()

    rank_zero_info("=" * 50 + "END PREDICT" + "=" * 50)
    rank_zero_info("=" * 50 + "END %s PREDICT" % stage.upper() + "=" * 50)


def predict(
    stage: str,
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    backend: str = "NCCL",
    launcher: str = None,
    pipeline_test: bool = False,
    ckpt: str = None,
    args_env: list = None,
    enable_tracking: bool = False,
):
    """Predict  function.

    Args:
        stage: The training stage, you should define {stage}_predictor
               in your config.
        config: Config file path.
        device_ids: GPU device ids like '0,1,2,3', will override
                    `device_ids` in config.
        dist_url: Dist url for init process, such as tcp://localhost:8000.
        backend: Dist url for init process.
        launcher: Job launcher for multi machines.
        ckpt: Checkpoint path used to predict.
        args_env: The args will be set in env.
    """
    if args_env:
        setup_args_env(args_env)
    setup_hat_env(stage, pipeline_test)
    config_info = Config.fromfile(config)
    # check config version
    config_version = config_info.get("VERSION", None)
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
    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids
    if ids is None or ids == -1:
        assert backend == "GLOO", f"backend should be GLOO, but get {backend}"
    num_processes = config_info.get("num_processes", None)

    predictor_cfg = config_info[f"{stage}_predictor"]

    launch = build_launcher(predictor_cfg)
    launch(
        predict_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        num_processes=num_processes,
        backend=backend,
        args=(stage, config, ckpt),
    )


if __name__ == "__main__":

    try:
        args, args_env = parse_args()
        predict(
            stage=args.stage,
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            backend=args.backend,
            launcher=args.launcher,
            pipeline_test=args.pipeline_test,
            ckpt=args.ckpt,
            args_env=args_env,
        )
    except Exception as e:
        logger.error("predict failed! " + str(e))
        raise e
