"""align bpu validation tools, Only support int-infer"""
import argparse

import horizon_plugin_pytorch as horizon
import torch

from hat.data.collates.collates import collate_2d
from hat.engine.processors import BasicBatchProcessor
from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        required=False,
        default=1,
        help="Reference device, only support single device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=1,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="QAT checkpoint for int-infer.",
    )
    return parser.parse_args()


def _modify_config(cfg, num_workers=None, device=None, pretrained_ckpt=None):

    assert "data_loader" in cfg

    if "sampler" in cfg["data_loader"]:
        cfg["data_loader"].pop("sampler")
    cfg["data_loader"]["shuffle"] = False
    if num_workers is not None:
        cfg["data_loader"]["num_workers"] = num_workers
    cfg["data_loader"]["batch_size"] = 1
    cfg["data_loader"]["shuffle"] = False
    if "collate_fn" not in cfg["data_loader"]:
        cfg["data_loader"]["collate_fn"] = collate_2d
    cfg["device"] = device
    cfg["batch_processor"] = BasicBatchProcessor(need_grad_update=False)
    if pretrained_ckpt is not None:
        cfg["model_convert_pipeline"]["converters"][1][
            "checkpoint_path"
        ] = pretrained_ckpt
    return cfg


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(int(args.device_id))
    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_align_bpu_validation")
    horizon.march.set_march(config.get("march"))

    align_bpu_cfg = config.get("align_bpu_predictor")
    align_bpu_cfg = _modify_config(
        cfg=align_bpu_cfg,
        num_workers=args.num_workers,
        device=args.device_id,
        pretrained_ckpt=args.pretrained_ckpt,
    )

    predictor = build_from_registry(align_bpu_cfg)

    predictor.fit()
