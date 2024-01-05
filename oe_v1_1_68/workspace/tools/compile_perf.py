# Copyright (c) Horizon Robotics. All rights reserved.
import argparse
import copy
import logging
import os
import pprint
import warnings
from distutils.version import LooseVersion
from typing import Optional

import horizon_plugin_pytorch as horizon
import torch
from hbdk import hbir_base
from horizon_plugin_pytorch.quantization import perf_model

from hat.registry import RegistryContext, build_from_registry
from hat.utils.apply_func import _as_list
from hat.utils.checkpoint import load_state_dict
from hat.utils.config import Config, ConfigVersion
from hat.utils.hash import generate_sha256_file
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compile_then_perf(
    cfg_file: str,
    out_dir: Optional[str] = None,
    opt: Optional[str] = None,
    jobs: Optional[int] = 0,
    ckpt: Optional[str] = None,
    save_qresults: bool = False,
):  # noqa: D205,D400
    """Compile deploy_model of stage `int_infer` then test performance of it,
    `.hbm` and other performance file like `.json` will save.

    Args:
        cfg_file: Config file name.
        out_dir: Directory to hold performance files.
            If not None, will override `config.compile_cfg["out_dir"]`.
            If None, no-op.
        opt:
            If not None, will override `config.compile_cfg["opt"]`.
            If None, no-op.
        jobs: Number of threads launched during compiler optimization.
            Default 0 means to use all available hardware concurrency.
        ckpt: ckpt path for int_infer stage.
        save_qresults: If set True, each module result in quantized
            model will be saved in `plugin_quantized_result.npy` with
            `allow_pickle=True` to compare with hbdk parser, which MUST also be
            set True when loading the file by numpy.
    """
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

    # check BPU march
    march = cfg.get("march", None)
    if march:
        horizon.march.set_march(march)
    else:
        raise ValueError("`march` in config can not be None.")

    with RegistryContext():
        int_infer_trainer = cfg["int_infer_trainer"]
        int_infer_trainer = build_from_registry(int_infer_trainer)

    # device: cpu
    device = torch.device("cpu")
    deploy_model = int_infer_trainer.model
    deploy_model.to(device)
    deploy_inputs = cfg.deploy_inputs
    if ckpt is not None:
        logger.warning("Make sure ckpt is from int_infer stage")
        load_state_dict(
            deploy_model,
            path_or_dict=ckpt,
            map_location=device,
        )

    # have a test
    deploy_model.eval()
    deploy_model(deploy_inputs)

    # override default compile config
    compile_cfg = (
        copy.deepcopy(cfg.compile_cfg) if cfg.compile_cfg is not None else {}
    )

    if out_dir is not None:
        compile_cfg["out_dir"] = out_dir
        compile_cfg["hbm"] = os.path.join(compile_cfg["out_dir"], "model.hbm")
    if opt is not None:
        compile_cfg["opt"] = opt

    if compile_cfg["out_dir"] is None:
        compile_cfg["out_dir"] = "."
        compile_cfg["hbm"] = os.path.join(compile_cfg["out_dir"], "model.hbm")
    if compile_cfg["hbm"] is None:
        cfg_name = os.path.splitext(os.path.basename(cfg_file))[0]
        compile_cfg["hbm"] = os.path.join(
            compile_cfg["out_dir"], "%s-deploy_model.hbm" % cfg_name
        )
    compile_cfg["jobs"] = jobs
    qsave = compile_cfg.pop("save_qresults", False)
    qsave = save_qresults if not qsave else qsave

    if not os.path.exists(compile_cfg["out_dir"]):
        os.makedirs(compile_cfg["out_dir"])

    logger.info("Compile config:\n" + pprint.pformat(compile_cfg))

    # compile, perf
    # wrap dict, tensor as list
    example_inputs = tuple(_as_list(deploy_inputs))
    hbir_base.CleanUpContext()
    result = perf_model(
        module=deploy_model.eval(),
        example_inputs=example_inputs,
        **compile_cfg,
    )
    if isinstance(result, dict):
        hashed_hbm_file = generate_sha256_file(
            compile_cfg["hbm"], remove_old=True
        )
        logger.info("Perf details:\n" + pprint.pformat(result))
        logger.info(
            format_msg("Compiled model: %s" % hashed_hbm_file, MSGColor.GREEN)
        )
        logger.info(
            format_msg(
                "Performance results saved at: %s" % compile_cfg["out_dir"],
                MSGColor.GREEN,
            )
        )
    else:
        raise RuntimeError(
            "Compile or perf deploy_model fail in stage int_infer"
        )

    if qsave:
        logger.info("Save quantized results...")
        if LooseVersion(horizon.__version__) < "1.0.1":
            raise ImportError(
                "If you want to save quantized results to compare with hbdk "
                "parser, please make sure the version of your plugin not less"
                " than 1.0.1"
            )

        horizon.utils.quant_profiler.script_profile(
            deploy_model,
            deploy_inputs,
            compile_cfg["out_dir"],
        )


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
        "--out-dir",
        type=str,
        required=False,
        default=None,
        help="directory to hold perf results like `.hbm`, `.json`, will "
        'override config.compile_cfg["out_dir"]',
    )
    parser.add_argument(
        "--opt",
        type=str,
        required=False,
        default=None,
        help='optimization options, will override config.compile_cfg["opt"], '
        "available options are O0, O1, O2, O3, ddr, fast, balance. ",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        required=False,
        default=4,
        help="number of threads launched during compiler optimization."
        " Default 0 means to use all available hardware concurrency.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path to load for int_infer stage",
    )
    parser.add_argument(
        "--save_qresults",
        type=int,
        required=False,
        default=0,
        help="whether to save the quantized model results in npy to compare"
        " with compiler. Default 0 means not save",
        choices=[0, 1],
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    compile_then_perf(
        cfg_file=args.config,
        out_dir=args.out_dir,
        opt=args.opt,
        jobs=args.jobs,
        ckpt=args.ckpt,
        save_qresults=bool(args.save_qresults),
    )
