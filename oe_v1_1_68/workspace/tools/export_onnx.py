import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.utils.onnx_helper import (
    export_quantized_onnx,
    export_to_onnx,
)

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
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(args.config)

    logger.info("=" * 50 + "BEGIN EXPORT ONNX" + "=" * 50)

    if "march" not in cfg:
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    with RegistryContext():
        onnx_solver = build_from_registry(cfg.onnx_cfg)
        model = onnx_solver["model"].eval()
        pipeline = onnx_solver.get("model_convert_pipeline")
        stage = onnx_solver["stage"]
        
        # if pipeline is not None:
        #     model = pipeline(model)
        # else:
        #     logger.warning(
        #         format_msg(
        #             f"not define model_convert_pipeline for {stage} stage "
        #             f"model, will directly export the model to onnx...",
        #             MSGColor.RED,
        #         )
        #     )

    example_input = onnx_solver.get("inputs", cfg.deploy_inputs)
    out_dir = onnx_solver.get("out_dir", cfg.get("ckpt_dir", "."))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file_path = os.path.join(out_dir, stage + ".onnx")
    kwargs = onnx_solver.get("kwargs", {})

    logger.info("will export {} model to onnx...".format(stage))
    if stage == "int_infer":
        # If a dictionary is the last element of the args tuple, it will be
        # interpreted as containing named arguments. In order to pass a dict as
        # the last non-keyword arg, provide an empty dict as the last element
        # of the args tuple.
        export_quantized_onnx(model, (example_input, {}), file_path, **kwargs)
    else:
        export_to_onnx(model, (example_input, {}), file_path, **kwargs)

    logger.info("=" * 50 + "END ONNX" + "=" * 50)
