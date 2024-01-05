"""align bpu validation tools, Only support int-infer"""
import argparse
import os
import pathlib
from functools import partial

import horizon_plugin_pytorch as horizon
import torch
import torchvision

from hat.registry import build_from_registry
from hat.utils.apply_func import to_cuda
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
        default=0,
        help="Reference device, only support single device.",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model."
    )
    parser.add_argument(
        "--model-inputs", type=str, default=None, help="model input."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="save path for visualize output.",
    )
    return parser.parse_args()


def defualt_prepare_inputs(infer_inputs):

    return [infer_inputs]


def _analyze_inputs(inputs):

    inputs_list = inputs.split(",")
    model_inputs = {}
    for each_input in inputs_list:
        name, values = each_input.split(":")
        model_inputs[name.strip()] = values.strip()

    return model_inputs


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.set_device(int(args.device_id))
    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_infer_viz")
    horizon.march.set_march(config.get("march"))

    infer_cfg = config.get("infer_cfg")

    # get model inputs and process
    prepare_inputs = infer_cfg.get("prepare_inputs", defualt_prepare_inputs)
    process_inputs = infer_cfg.get("process_inputs")
    process_outputs = infer_cfg.get("process_outputs")
    if args.model_inputs is not None:
        infer_inputs = _analyze_inputs(args.model_inputs)
    else:
        infer_inputs = infer_cfg.get("infer_inputs")

    prepared_inputs = prepare_inputs(infer_inputs)

    # build data transforms
    transforms = infer_cfg.get("transforms", None)

    if transforms is not None:
        transforms = build_from_registry(transforms)
        transforms = torchvision.transforms.Compose(transforms)

    # build model and load ckpt
    model = build_from_registry(infer_cfg.get("model"))
    model.eval()
    model_convert_pipeline = infer_cfg.get("model_convert_pipeline")
    if args.pretrained_ckpt is not None:
        model_convert_pipeline["converters"][1][
            "checkpoint_path"
        ] = args.pretrained_ckpt
    model_convert_pipeline = build_from_registry(model_convert_pipeline)
    quantized_model = model_convert_pipeline(model)

    quantized_model.cuda(args.device_id)

    viz_func = build_from_registry(infer_cfg.get("viz_func"))

    if isinstance(viz_func, list):
        for i, f in enumerate(viz_func):
            viz_func[i] = partial(f, save_path=args.save_path)
    else:
        viz_func = partial(viz_func, save_path=args.save_path)

    for prepared_input in prepared_inputs:
        model_input, vis_inputs = process_inputs(prepared_input, transforms)
        model_input = to_cuda(
            model_input, device=args.device_id, non_blocking=True
        )
        with torch.no_grad():
            model_outputs = quantized_model(model_input)
        outputs = process_outputs(model_outputs, viz_func, vis_inputs)
        if outputs is not None:
            print(outputs)
